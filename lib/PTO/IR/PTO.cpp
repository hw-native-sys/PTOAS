// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTO.cpp - PTO Dialect ----------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/PTOSyncUtils.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;

// Forward declarations for custom shape/type printers used by tensor_view and
// partition_tensor_view.
namespace mlir {
namespace pto {
static LogicalResult parseShapeAndElem(AsmParser &parser,
                                       SmallVectorImpl<int64_t> &shape,
                                       Type &elementType,
                                       bool allowDynamic = true);
static void printShapeAndElem(AsmPrinter &printer,
                              ArrayRef<int64_t> shape,
                              Type elementType);
} // namespace pto
} // namespace mlir

// =============================================================================
// TileBufType 的自定义 Shape 解析与打印函数
// =============================================================================

// 解析逻辑：解析形如 "32x32" 的维度列表
[[maybe_unused]] static ParseResult parseShape(AsmParser &parser, SmallVectorImpl<int64_t> &shape) {
  // parseDimensionList 会解析 "dim x dim x ...", 遇到无法解析为维度的字符停止
  // 参数 allowDynamic=true (允许 ?), withTrailingX=false (不吞掉末尾的 x)
  if (parser.parseDimensionList(shape, /*allowDynamic=*/true, /*withTrailingX=*/false))
    return failure();
  return success();
}

// 打印逻辑：打印形如 "32x32" 的维度列表
[[maybe_unused]] static void printShape(AsmPrinter &printer, ArrayRef<int64_t> shape) {
  for (auto it = shape.begin(); it != shape.end(); ++it) {
    if (it != shape.begin()) printer << "x"; // 维度间的分隔符
    if (*it == ShapedType::kDynamic)
      printer << "?";
    else
      printer << *it;
  }
  // 注意：我们不在这里打印末尾的 'x'，因为 assemblyFormat 中已经写了 `x` $elementType
}

static std::optional<pto::AddressSpace> getPTOMemorySpaceEnum(Type ty);
enum class VerifierTargetArch {
  A2A3,
  A5,
};
static VerifierTargetArch getVerifierTargetArch(Operation *op);
static std::optional<StringRef> getVerifierArchName(Operation *op);
static bool isSupportedVecElemType(Type ty, bool allowBf16 = true,
                                   bool allowInt8 = true);
static bool isSupportedLoadStoreElemTypeA2A3(Type ty);
static bool isSupportedGatherElemTypeA2A3(Type ty);
static bool isSupportedGatherElemTypeA5(Type ty);
static bool isA5TLoadStoreTransferElemType(Type ty);
static bool isA5AccStorePreQuantDstType(Type srcElem, Type dstElem);
static bool isA5LowPrecisionTCvtPair(Type srcElem, Type dstElem);
static bool isA5SupportedTCvtPair(Type srcElem, Type dstElem);
static ParseResult parseSyncEventOpCommon(OpAsmParser &parser,
                                          OperationState &result,
                                          StringAttr pipeAttrName,
                                          StringAttr eventIdAttrName);
static void printSyncEventOpCommon(OpAsmPrinter &p, Operation *op,
                                   PipeAttr pipeAttr, IntegerAttr eventAttr,
                                   Value eventDyn, StringRef pipeAttrName,
                                   StringRef eventIdAttrName);
static bool isTileLikeType(Type ty);
static SmallVector<int64_t, 4> getShapeVec(Type ty);
static SmallVector<int64_t, 4> getValidShapeVec(Type ty);
static SmallVector<int64_t, 4> getValidShapeVec(Value value);
static bool isKnownZeroOrUnitExtent(int64_t value);
static bool isByteIntegerType(Type ty);
static LogicalResult verifyTileBufCommon(Operation *op, Type ty, StringRef name,
                                         bool allowLowPrecision = false);
static LogicalResult verifyTileBufSameElemType(Operation *op, Type lhs, Type rhs,
                                               StringRef lhsName,
                                               StringRef rhsName);
static LogicalResult verifyTileBufSameLogicalExtent(Operation *op, Type lhs,
                                                    Type rhs, StringRef lhsName,
                                                    StringRef rhsName,
                                                    bool compareValidShape);

static LogicalResult verifyTileBufSameValidShape(Operation *op, Type lhs, Type rhs,
                                                 StringRef lhsName, StringRef rhsName);
static LogicalResult verifyVecTileCommon(Operation *op, Type ty, StringRef name);
static LogicalResult verifyVecTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name);
static LogicalResult verifyVecTileCommonA5(Operation *op, Type ty,
                                           StringRef name);
static LogicalResult verifyVecTileUnaryOp(Operation *op, Type srcTy, Type dstTy,
                                          StringRef srcName = "src",
                                          StringRef dstName = "dst",
                                          bool allowBf16 = true,
                                          bool allowInt8 = true);
static LogicalResult verifyAccTileCommon(Operation *op, Type ty, StringRef name);
static LogicalResult verifyAccTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name);
static LogicalResult verifyAccTileCommonA5(Operation *op, Type ty,
                                           StringRef name);
static LogicalResult verifyMatTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                           Type dstTy);
static LogicalResult verifyMatTileOperandsA2A3(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy);
static LogicalResult verifyMatTileOperandsA5(Operation *op, Type lhsTy,
                                             Type rhsTy, Type dstTy);
static LogicalResult verifyGemvTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                            Type dstTy);
static LogicalResult verifyAsyncFlatContiguous1DGMViewLike(Operation *op,
                                                           Value value,
                                                           StringRef name);
static LogicalResult verifyGemvTileOperandsA2A3(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy);
static LogicalResult verifyGemvTileOperandsA5(Operation *op, Type lhsTy,
                                              Type rhsTy, Type dstTy);
static LogicalResult verifyMatBiasTile(Operation *op, Type biasTy, Type dstTy,
                                       bool requireFloatBias = false);
static LogicalResult verifyMatBiasTileA2A3(Operation *op, Type biasTy, Type dstTy,
                                           bool requireFloatBias = false);
static LogicalResult verifyMatBiasTileA5(Operation *op, Type biasTy, Type dstTy,
                                         bool requireFloatBias = false);
static LogicalResult verifyMatmulTypeTriple(Operation *op, Type lhsElemTy,
                                            Type rhsElemTy, Type dstElemTy);
static std::optional<pto::Layout> getLogicalViewLayout(Value value);
static std::optional<pto::Layout> getTileBufLogicalLayout(pto::TileBufType type);
static std::optional<int64_t> getConstantIntegerValue(Value value);
static LogicalResult verifyPartialValidPattern(Operation *op, Type src0Ty,
                                               Type src1Ty, Type dstTy);
static Type getElemTy(Type ty);
static FailureOr<Type>
verifyMatchingRowMajorBinaryTileOpCommon(Operation *op, Type src0Ty,
                                         Type src1Ty, Type dstTy);
static FailureOr<Type>
verifyNumericScalarTileOpCommon(Operation *op, Type srcTy, Type dstTy,
                                Type scalarTy, bool requireValidRowsEqual);
static FailureOr<Type>
verifyShiftLikeBinaryTileOpCommon(Operation *op, Type src0Ty, Type src1Ty,
                                  Type dstTy);
static LogicalResult verifyArithmeticElemTypeForArch(
    Operation *op, Type elemTy, PTOArch targetArch, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error);
static bool isRowMajorTileBuf(Type ty);
static ParseResult parseLegacyOrAttrPipe(OpAsmParser &parser, PipeAttr &attr);
static ParseResult parseLegacyOrAttrEvent(OpAsmParser &parser, EventAttr &attr);
static ParseResult parseI32LiteralAttr(OpAsmParser &parser, IntegerAttr &attr);

#define GET_ENUM_CLASSES
#include "PTO/IR/PTOEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "PTO/IR/PTOTypeDefs.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "PTO/IR/PTOAttrs.cpp.inc"

#include "PTO/IR/PTODialect.cpp.inc"

[[maybe_unused]] static LogicalResult parseShapeAndElemStable(mlir::AsmParser &parser,
                                             llvm::SmallVectorImpl<int64_t> &shape,
                                             mlir::Type &elementType) {
  if (failed(parser.parseLess()))
    return failure();

  if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/true)))
    return failure();

  if (failed(parser.parseType(elementType)))
    return failure();

  if (failed(parser.parseGreater()))
    return failure();

  return success();
}

static int64_t getPTOTypeRank(Type type) {
  // 1. 处理标准的 MLIR 类型 (MemRef, Tensor, Vector)
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    if (shapedTy.hasRank())
      return shapedTy.getRank();
    return -1; // Unranked type
  }
  
  // 2. 处理 PTO 自定义类型
  if (auto tvTy = dyn_cast<pto::TensorViewType>(type))
    return tvTy.getRank();

  if (auto tileTy = dyn_cast<pto::TileType>(type))
    return tileTy.getRank();
    
  if (auto tileViewTy = dyn_cast<pto::PartitionTensorViewType>(type))
    return tileViewTy.getRank();

  if (auto tileBufTy = dyn_cast<pto::TileBufType>(type))
    return tileBufTy.getRank();

  // 3. 不支持的类型
  return -1;
}

static bool isGmAddressSpaceAttr(Attribute memorySpace) {
  if (!memorySpace)
    return true;
  if (auto addr = mlir::dyn_cast<pto::AddressSpaceAttr>(memorySpace))
    return addr.getAddressSpace() == pto::AddressSpace::GM;
  if (auto intAttr = mlir::dyn_cast<IntegerAttr>(memorySpace))
    return intAttr.getInt() == 0;
  return false;
}

PTOArch mlir::pto::getTargetArch(ModuleOp module) {
  if (!module)
    return PTOArch::A3;

  auto arch = module->getAttrOfType<StringAttr>(kPTOTargetArchAttrName);
  if (arch && arch.getValue().equals_insensitive("a5"))
    return PTOArch::A5;
  return PTOArch::A3;
}

PTOArch mlir::pto::getTargetArch(Operation *op) {
  if (!op)
    return PTOArch::A3;
  return getTargetArch(op->getParentOfType<ModuleOp>());
}

bool mlir::pto::isTargetArchA3(ModuleOp module) {
  return getTargetArch(module) == PTOArch::A3;
}

bool mlir::pto::isTargetArchA5(ModuleOp module) {
  return getTargetArch(module) == PTOArch::A5;
}

bool mlir::pto::isTargetArchA3(Operation *op) {
  return getTargetArch(op) == PTOArch::A3;
}

bool mlir::pto::isTargetArchA5(Operation *op) {
  return getTargetArch(op) == PTOArch::A5;
}

static llvm::TypeSize getOneByteTypeSize() {
  return llvm::TypeSize::getFixed(8);
}

llvm::TypeSize mlir::pto::HiF8Type::getTypeSizeInBits(
    const DataLayout &, DataLayoutEntryListRef) const {
  return getOneByteTypeSize();
}

uint64_t mlir::pto::HiF8Type::getABIAlignment(const DataLayout &,
                                              DataLayoutEntryListRef) const {
  return 1;
}

uint64_t mlir::pto::HiF8Type::getPreferredAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 1;
}

static llvm::TypeSize getTwoByteTypeSize() {
  return llvm::TypeSize::getFixed(16);
}

llvm::TypeSize mlir::pto::HiF8x2Type::getTypeSizeInBits(
    const DataLayout &, DataLayoutEntryListRef) const {
  return getTwoByteTypeSize();
}

uint64_t mlir::pto::HiF8x2Type::getABIAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 2;
}

uint64_t mlir::pto::HiF8x2Type::getPreferredAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 2;
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

static VerifierTargetArch getVerifierTargetArch(Operation *op) {
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
  if (!module)
    return std::nullopt;
  if (auto arch = module->getAttrOfType<StringAttr>(kPTOTargetArchAttrName))
    return arch.getValue();
  return std::nullopt;
}

static bool shouldBypassDecodedMemrefVerifier(Operation *op) {
  if (!op)
    return false;
  for (Value operand : op->getOperands()) {
    if (isa<MemRefType>(operand.getType()))
      return true;
    if (operand.getDefiningOp<pto::BindTileOp>())
      return true;
  }
  return false;
}

static SmallVector<int64_t, 4> canonicalizeTileBufValidShape(ArrayRef<int64_t> validShape) {
  SmallVector<int64_t, 4> canonical;
  canonical.reserve(validShape.size());
  for (int64_t dim : validShape)
    canonical.push_back(dim < 0 ? ShapedType::kDynamic : dim);
  return canonical;
}

template <typename FnA2A3, typename FnA5>
static LogicalResult dispatchVerifierByArch(Operation *op, FnA2A3 &&verifyA2A3,
                                            FnA5 &&verifyA5) {
  if (shouldBypassDecodedMemrefVerifier(op))
    return success();
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
  default:
    return {};
  }
}

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
  if (auto ft = dyn_cast<FloatType>(ty)) {
    unsigned width = ft.getWidth();
    return width == 8;
  }
  if (auto it = dyn_cast<IntegerType>(ty))
    return it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32;
  return false;
}

static std::optional<mlir::pto::Layout>
inferLayout(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
            unsigned elemBytes) {
  if (shape.size() != strides.size() || elemBytes == 0)
    return std::nullopt;

  // NZ / fractal: rank>=5, check middle dims (sh3/sh4/sh5 per spec)
  if (shape.size() >= 5) {
    int64_t sh3 = shape[2], sh4 = shape[3], sh5 = shape[4];
    int64_t st4 = strides[3], st5 = strides[4];
    bool alignMatch = (sh3 == 16) && (sh3 * sh4 * elemBytes == 512);
    bool strideMatch = (st5 == 1) && (st4 == sh5);
    if (alignMatch && strideMatch)
      return mlir::pto::Layout::NZ;
  }

  // ND: row-major contiguous
  bool isRowMajor = true;
  for (int i = 0, e = (int)shape.size() - 1; i < e; ++i) {
    if (strides[i] != strides[i + 1] * shape[i + 1]) {
      isRowMajor = false;
      break;
    }
  }
  if (isRowMajor && strides.back() == 1)
    return mlir::pto::Layout::ND;

  // DN: col-major
  bool isColMajor = true;
  for (int i = 0, e = (int)shape.size() - 1; i < e; ++i) {
    if (strides[i + 1] != strides[i] * shape[i]) {
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

static LogicalResult verifyTRowReductionNoTmpCommon(Operation *op, Type srcTy,
                                                    Type dstTy,
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
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, tmpTy, "src", "tmp")))
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

static LogicalResult verifyTRowArgReductionCommon(Operation *op, Type srcTy,
                                                  Type tmpTy, Type dstTy) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, tmpTy, "src", "tmp")))
    return failure();
  // Arg reductions still emit a real per-row index, so the empty-region no-op
  // marker is not accepted here (unlike plain trowmax/trowsum above).
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

static LogicalResult verifyNDStyleVecTile(Operation *op, Type ty, StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name)))
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

static LogicalResult verifyColReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy,
                                                   bool requireNonZeroSrc) {
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError("expects src and dst to have rank-2 valid_shape");
  // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. The op
  // writes no elements; accept and skip the non-empty constraints. One-sided
  // empties still fall through. See pto-isa#143 for hardware Rv=0 no-op.
  // Col arg reductions (tcolargmax/tcolargmin) never reach this point with a
  // 0x0 dst: verifyColArgReductionDstLayout enforces dst valid_shape[0] == 1
  // first, so they stay strict without needing a flag here (unlike the row
  // path, whose dst-layout check does not constrain valid).
  if (dstValid[0] == 0 && dstValid[1] == 0)
    return success();
  if (requireNonZeroSrc) {
    if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0)
      return op->emitOpError("expects src valid_shape[0] to be non-zero");
    if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0)
      return op->emitOpError("expects src valid_shape[1] to be non-zero");
  }
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1])
    return op->emitOpError("expects src and dst to have the same valid_shape[1]");
  return success();
}

static LogicalResult verifyColArgReductionDstLayout(Operation *op, Type ty,
                                                    StringRef name) {
  if (failed(verifyNDStyleVecTile(op, ty, name)))
    return failure();
  auto valid = getValidShapeVec(ty);
  if (valid.size() != 2)
    return op->emitOpError() << "expects " << name
                             << " to have rank-2 valid_shape";
  if (valid[0] != ShapedType::kDynamic && valid[0] != 1)
    return op->emitOpError() << "expects " << name
                             << " valid_shape[0] to be 1";
  return success();
}

static std::optional<int64_t> getConstantIntegerValue(Value value) {
  if (!value)
    return std::nullopt;
  if (auto arithCst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(arithCst.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

LogicalResult mlir::pto::MakeTensorViewOp::verify() {
  auto tvTy = dyn_cast<mlir::pto::TensorViewType>(getResult().getType());
  if (!tvTy)
    return emitOpError("result must be pto.tensor_view<...>");

  auto pty = dyn_cast<mlir::pto::PtrType>(getPtr().getType());
  if (!pty)
    return emitOpError("ptr operand must be !pto.ptr<...>");

  if (pty.getElementType() != tvTy.getElementType())
    return emitOpError() << "ptr element type must match tensor_view element type, but got ptr="
                         << pty.getElementType() << " view=" << tvTy.getElementType();

  int64_t rank = tvTy.getRank();

  if ((int64_t)getShape().size() != rank || (int64_t)getStrides().size() != rank)
    return emitOpError() << "shape/strides operand counts must match tensor_view rank="
                         << rank;

  // Detect dynamic shape/stride.
  bool hasDynamicShape = llvm::any_of(tvTy.getShape(), [](int64_t v) {
    return v == ShapedType::kDynamic;
  });
  bool hasDynamicStride = llvm::any_of(getStrides(), [](Value s) {
    return !getConstIndexValue(s).has_value();
  });

  auto layoutAttr = getLayoutAttr();

  // 1) Dynamic shape/stride without explicit layout: warn and keep going.
  if ((hasDynamicShape || hasDynamicStride) && !layoutAttr) {
    return success();
  }

  // 2) Static shape/stride with explicit layout: verify correctness.
  bool allStaticStride = true;
  SmallVector<int64_t> strideInts;
  strideInts.reserve(getStrides().size());
  for (Value s : getStrides()) {
    auto val = getConstIndexValue(s);
    if (!val) {
      allStaticStride = false;
      break;
    }
    strideInts.push_back(*val);
  }

  bool allStaticShape =
      llvm::none_of(tvTy.getShape(), [](int64_t v) { return v == ShapedType::kDynamic; });

  if (layoutAttr && allStaticShape && allStaticStride) {
    SmallVector<int64_t> shapeInts(tvTy.getShape().begin(), tvTy.getShape().end());
    if (auto inferred = inferLayout(shapeInts, strideInts,
                                    getElemByteSize(tvTy.getElementType()))) {
      (void)inferred;
    }
  }

  return success();
}

LogicalResult mlir::pto::PartitionViewOp::verify() {
  auto srcTy = dyn_cast<mlir::pto::TensorViewType>(getSource().getType());
  auto resTy = dyn_cast<mlir::pto::PartitionTensorViewType>(getResult().getType());
  if (!srcTy || !resTy)
    return emitOpError("expects tensor_view source and partition_tensor_view result");

  if (srcTy.getElementType() != resTy.getElementType())
    return emitOpError() << "element type mismatch between source and result: src="
                         << srcTy.getElementType() << " result="
                         << resTy.getElementType();

  int64_t srcRank = srcTy.getRank();
  if ((int64_t)getOffsets().size() != srcRank)
    return emitOpError() << "offset count (" << getOffsets().size()
                         << ") must match source rank (" << srcRank << ")";

  if ((int64_t)getSizes().size() != srcRank)
    return emitOpError() << "size count (" << getSizes().size()
                         << ") must match source rank (" << srcRank << ")";

  ArrayRef<int64_t> srcShape = srcTy.getShape();
  ArrayRef<int64_t> resShape = resTy.getShape();
  bool sameRank = resTy.getRank() == srcRank;

  for (int64_t i = 0; i < srcRank; ++i) {
    auto offVal = getConstIndexValue(getOffsets()[i]);
    auto sizeVal = getConstIndexValue(getSizes()[i]);

    if (offVal && *offVal < 0)
      return emitOpError() << "offset at dim " << i
                           << " must be non-negative, got " << *offVal;

    if (sizeVal && *sizeVal <= 0)
      return emitOpError() << "size at dim " << i
                           << " must be positive, got " << *sizeVal;

    if (sameRank && sizeVal) {
      int64_t resDim = resShape[i];
      if (resDim != ShapedType::kDynamic && *sizeVal != resDim)
        return emitOpError() << "size/result mismatch at dim " << i
                             << ": size operand=" << *sizeVal
                             << " result type dim=" << resDim;
    }

    int64_t srcDim = srcShape[i];
    if (srcDim == ShapedType::kDynamic)
      continue;

    if (sizeVal && *sizeVal > srcDim)
      return emitOpError() << "size at dim " << i << " (" << *sizeVal
                           << ") exceeds static source dim (" << srcDim << ")";

    if (offVal && sizeVal && (*offVal + *sizeVal > srcDim))
      return emitOpError() << "offset+size at dim " << i << " ("
                           << (*offVal + *sizeVal)
                           << ") exceeds static source dim (" << srcDim << ")";
  }

  return success();
}

LogicalResult mlir::pto::AddPtrOp::verify() {
  Value ptr = getOperation()->getOperand(0);
  Value result = getOperation()->getResult(0);

  auto ptrTy = dyn_cast<mlir::pto::PtrType>(ptr.getType());
  if (!ptrTy)
    return emitOpError("ptr operand must be !pto.ptr<...>");

  auto resTy = dyn_cast<mlir::pto::PtrType>(result.getType());
  if (!resTy)
    return emitOpError("result must be !pto.ptr<...>");

  if (ptrTy != resTy)
    return emitOpError("result type must match ptr operand type");

  return success();
}

static LogicalResult verifyPtrLikeForAddressCast(Operation *op, Type type,
                                                 StringRef name) {
  if (isa<mlir::pto::PtrType>(type))
    return success();

  auto memTy = dyn_cast<MemRefType>(type);
  if (!memTy)
    return op->emitOpError()
           << "expects " << name << " to be !pto.ptr<...> or a GM memref";

  if (memTy.getRank() != 1)
    return op->emitOpError()
           << "expects lowered memref " << name << " to be rank-1";

  if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
    return op->emitOpError()
           << "expects lowered memref " << name << " to use GM address space";

  return success();
}

static Type getPointerLikeElementType(Type type) {
  if (auto ptrTy = dyn_cast<mlir::pto::PtrType>(type))
    return ptrTy.getElementType();
  if (auto memTy = dyn_cast<MemRefType>(type))
    return memTy.getElementType();
  return Type();
}

static bool isEmitCSupportedScalarType(Type type) {
  if (!type)
    return false;
  if (type.isF16() || type.isBF16() || type.isF32() || type.isF64())
    return true;
  if (auto intTy = dyn_cast<IntegerType>(type))
    return intTy.getWidth() == 8 || intTy.getWidth() == 16 ||
           intTy.getWidth() == 32 || intTy.getWidth() == 64;
  if (mlir::pto::isPTOFloat8Type(type))
    return true;
  if (isa<mlir::pto::HiF8Type, mlir::pto::F4E1M2x2Type,
          mlir::pto::F4E2M1x2Type>(type))
    return true;
  return false;
}

LogicalResult mlir::pto::PtrToIntOp::verify() {
  Type resultTy = getResult().getType();
  auto intTy = dyn_cast<IntegerType>(resultTy);
  if (!intTy || intTy.getWidth() != 64)
    return emitOpError("result must be i64");

  return verifyPtrLikeForAddressCast(getOperation(), getPtr().getType(),
                                     "ptr operand");
}

LogicalResult mlir::pto::IntToPtrOp::verify() {
  auto addrTy = dyn_cast<IntegerType>(getAddr().getType());
  if (!addrTy || addrTy.getWidth() != 64)
    return emitOpError("address operand must be i64");

  if (failed(verifyPtrLikeForAddressCast(getOperation(), getResult().getType(),
                                         "result")))
    return failure();

  Type dstElem = getPointerLikeElementType(getResult().getType());
  if (!isEmitCSupportedScalarType(dstElem))
    return emitOpError("result element type is not supported by EmitC: ")
           << dstElem;

  return success();
}

LogicalResult mlir::pto::LocalArrayGetOp::verify() {
  auto arrayTy = getArray().getType();
  int64_t rank = arrayTy.getRank();
  int64_t numIdx = static_cast<int64_t>(getIndices().size());
  if (numIdx != rank)
    return emitOpError() << "expects " << rank
                         << " indices for !pto.local_array of rank " << rank
                         << ", got " << numIdx;
  if (getResult().getType() != arrayTy.getElementType())
    return emitOpError()
           << "result type " << getResult().getType()
           << " does not match array element type "
           << arrayTy.getElementType();
  return success();
}

LogicalResult mlir::pto::LocalArraySetOp::verify() {
  auto arrayTy = getArray().getType();
  int64_t rank = arrayTy.getRank();
  int64_t numIdx = static_cast<int64_t>(getIndices().size());
  if (numIdx != rank)
    return emitOpError() << "expects " << rank
                         << " indices for !pto.local_array of rank " << rank
                         << ", got " << numIdx;
  if (getValue().getType() != arrayTy.getElementType())
    return emitOpError() << "value type " << getValue().getType()
                         << " does not match array element type "
                         << arrayTy.getElementType();
  return success();
}

LogicalResult mlir::pto::CastPtrOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();

  auto inputPtrType = dyn_cast<mlir::pto::PtrType>(inputType);
  auto resultPtrType = dyn_cast<mlir::pto::PtrType>(resultType);
  auto inputMemRefType = dyn_cast<BaseMemRefType>(inputType);
  bool inputIsInteger = isa<IntegerType>(inputType);
  bool resultIsInteger = isa<IntegerType>(resultType);

  if (!inputPtrType && !inputMemRefType && !inputIsInteger)
    return emitOpError("input must be an integer, memref, or !pto.ptr<...>");
  if (!resultPtrType && !resultIsInteger)
    return emitOpError("result must be an integer or !pto.ptr<...>");

  if (inputIsInteger && resultIsInteger)
    return emitOpError("integer-to-integer cast is not a ptr cast");

  if (inputMemRefType && resultIsInteger)
    return emitOpError("memref-to-integer cast is unsupported");

  if (inputMemRefType && resultPtrType) {
    auto memrefSpace = dyn_cast_or_null<mlir::pto::AddressSpaceAttr>(
        inputMemRefType.getMemorySpace());
    auto resultSpace = resultPtrType.getMemorySpace();
    if (memrefSpace && memrefSpace != resultSpace)
      return emitOpError("memref-to-ptr cast must stay within the same PTO memory space");
  }

  if (inputPtrType && resultPtrType &&
      inputPtrType.getMemorySpace() != resultPtrType.getMemorySpace()) {
    return emitOpError("ptr-to-ptr cast must stay within the same PTO memory space");
  }

  return success();
}




void PTODialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "PTO/IR/PTOTypeDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "PTO/IR/PTOOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "PTO/IR/PTOAttrs.cpp.inc"
      >();
}


AddressSpaceAttr mlir::pto::getPTOAddressSpaceAttr(Type type) {
  if (auto ptrType = dyn_cast<PtrType>(type))
    return ptrType.getMemorySpace();
  auto memRefType = dyn_cast<BaseMemRefType>(type);
  if (!memRefType)
    return {};
  auto scopeAttr = dyn_cast<AddressSpaceAttr>(memRefType.getMemorySpace());
  if (!scopeAttr)
    return {};
  return scopeAttr;
}

bool mlir::pto::isScalarPtrOrMemRef(Type type) {
  if (auto pty = dyn_cast<mlir::pto::PtrType>(type))
    return static_cast<bool>(pty);
  if (auto memTy = dyn_cast<MemRefType>(type))
    return isGmAddressSpaceAttr(memTy.getMemorySpace());
  return false;
}

bool mlir::pto::hasPTOKernelAttr(Operation *op) {
  return op && (op->hasAttr(kPTOKernelAttrName) ||
                op->hasAttr(kLegacyPTOAICoreAttrName));
}

bool mlir::pto::isPTOKernelFunction(func::FuncOp func) {
  return func && !func.isDeclaration() && hasPTOKernelAttr(func.getOperation());
}

bool mlir::pto::hasExplicitPTOEntryAttr(func::FuncOp func) {
  return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName));
}

static constexpr StringLiteral kEffectivePTOEntryAttrName =
    "pto.internal.entry";

static SmallVector<func::FuncOp> getPTOFunctionDefinitions(ModuleOp module) {
  SmallVector<func::FuncOp> defs;
  if (!module)
    return defs;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (!func.isDeclaration())
      defs.push_back(func);
  }
  return defs;
}

bool mlir::pto::isPTOEntryFunction(func::FuncOp func) {
  if (!func || func.isDeclaration())
    return false;
  if (auto attr = func->getAttrOfType<BoolAttr>(kEffectivePTOEntryAttrName))
    return attr.getValue();
  if (hasExplicitPTOEntryAttr(func))
    return true;

  ModuleOp module = func->getParentOfType<ModuleOp>();
  if (!module)
    return false;
  SmallVector<func::FuncOp> defs = getPTOFunctionDefinitions(module);
  return defs.size() == 1 && defs.front() == func;
}

LogicalResult mlir::pto::validatePTOEntryFunctions(ModuleOp module) {
  if (!module)
    return success();

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!hasExplicitPTOEntryAttr(func))
      continue;
    if (func.isDeclaration()) {
      return func.emitOpError()
             << "`" << kPTOEntryAttrName
             << "` is only valid on function definitions";
    }
  }

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!isPTOEntryFunction(func))
      continue;
    if (func.getFunctionType().getNumResults() != 0) {
      return func.emitOpError()
             << "PTO entry functions must return void";
    }
  }
  return success();
}

void mlir::pto::annotatePTOEntryFunctions(ModuleOp module) {
  if (!module)
    return;

  SmallVector<func::FuncOp> defs = getPTOFunctionDefinitions(module);
  for (auto func : module.getOps<func::FuncOp>())
    func->removeAttr(kEffectivePTOEntryAttrName);

  if (defs.empty())
    return;
  if (defs.size() == 1) {
    defs.front()->setAttr(kEffectivePTOEntryAttrName,
                          BoolAttr::get(module.getContext(), true));
    return;
  }

  for (auto func : defs) {
    func->setAttr(kEffectivePTOEntryAttrName,
                  BoolAttr::get(module.getContext(),
                                hasExplicitPTOEntryAttr(func)));
  }
}

//===----------------------------------------------------------------------===//
// PTO Load/Store/Addf (non-DPS polymorphic) verification + inference.
//  - If operands are memref/tensor: verify strictly.
//  - Otherwise (tile_view/tile etc): accept (so old IR can still parse).
//===----------------------------------------------------------------------===//

[[maybe_unused]] static LogicalResult verifyMemrefToTensorLoad(Operation *op, Value src, Value res) {
  auto mr = dyn_cast<MemRefType>(src.getType());
  auto rt = dyn_cast<RankedTensorType>(res.getType());
  if (!mr)
    return success(); // non-memref case: don't block old IR
  if (!rt)
    return op->emitOpError("when src is memref, result must be ranked tensor");

  if (mr.getElementType() != rt.getElementType())
    return op->emitOpError() << "memref/tensor element type mismatch: memref="
                             << mr.getElementType() << " tensor=" << rt.getElementType();

  if (mr.getRank() != rt.getRank())
    return op->emitOpError() << "rank mismatch: memref rank=" << mr.getRank()
                             << " tensor rank=" << rt.getRank();

  if (mr.hasStaticShape()) {
    if (!rt.hasStaticShape())
      return op->emitOpError("memref has static shape but result tensor is not static");
    if (mr.getShape() != rt.getShape())
      return op->emitOpError() << "shape mismatch: memref=" << mr << " tensor=" << rt;
  } else {
    // For dynamic memref dims: if tensor dim is static, allow it; if it's dynamic too, also fine.
    // We only reject when a memref static dim conflicts with tensor static dim.
    for (int64_t i = 0; i < mr.getRank(); ++i) {
      int64_t md = mr.getDimSize(i);
      int64_t td = rt.getDimSize(i);
      if (md != ShapedType::kDynamic && td != ShapedType::kDynamic && md != td)
        return op->emitOpError() << "dim mismatch at " << i << ": memref=" << md << " tensor=" << td;
    }
  }
  return success();
}

[[maybe_unused]] static LogicalResult verifyMemrefTensorStore(Operation *op, Value dst, Value src) {
  auto mr = dyn_cast<MemRefType>(dst.getType());
  if (!mr)
    return success(); // non-memref case: old tile IR allowed
  auto rt = dyn_cast<RankedTensorType>(src.getType());
  if (!rt)
    return op->emitOpError("when dst is memref, src must be ranked tensor");

  if (mr.getElementType() != rt.getElementType())
    return op->emitOpError() << "memref/tensor element type mismatch: memref="
                             << mr.getElementType() << " tensor=" << rt.getElementType();

  if (mr.getRank() != rt.getRank())
    return op->emitOpError() << "rank mismatch: memref rank=" << mr.getRank()
                             << " tensor rank=" << rt.getRank();

  for (int64_t i = 0; i < mr.getRank(); ++i) {
    int64_t md = mr.getDimSize(i);
    int64_t td = rt.getDimSize(i);
    if (md != ShapedType::kDynamic && td != ShapedType::kDynamic && md != td)
      return op->emitOpError() << "dim mismatch at " << i << ": memref=" << md << " tensor=" << td;
  }
  return success();
}

LogicalResult AllocTileOp::verify() {
  auto ty = getResult().getType(); // TileBufType

  if (failed(verifyTileBufLayoutConstraints(*this, ty, "result")))
    return failure();

  // op 上有没有传 operands
  bool hasVR = getValidRow() != nullptr;
  bool hasVC = getValidCol() != nullptr;

  // type 上的 validShape
  auto vs = ty.getValidShape();
  if (vs.size() != 2)
    return emitOpError("result tile_buf must have rank-2 validShape");

  // TileBuf valid dims use a negative sentinel (e.g. '?' / -1). Be robust to
  // any negative value (some code may materialize MLIR dynamic sentinels).
  bool needVR = (vs[0] < 0);
  bool needVC = (vs[1] < 0);

  // 你要求的：v_row=?, v_col=? 时必须同时给两个
  // （这条规则由下面两句自然实现）
  if (hasVR != needVR)
    return emitOpError() << "valid_row operand "
                         << (needVR ? "is required" : "must be absent")
                         << " because result type v_row is "
                         << (needVR ? "?" : std::to_string(vs[0]));

  if (hasVC != needVC)
    return emitOpError() << "valid_col operand "
                         << (needVC ? "is required" : "must be absent")
                         << " because result type v_col is "
                         << (needVC ? "?" : std::to_string(vs[1]));

  return success();
}

LogicalResult MaterializeTileOp::verify() {
  auto sourceTy = cast<MemRefType>(getSource().getType());
  auto resultTy = cast<TileBufType>(getResult().getType());

  if (sourceTy.getRank() != 2)
    return emitOpError("source memref must be rank-2 to materialize a tile handle");
  if (resultTy.getRank() != 2)
    return emitOpError("result tile_buf must be rank-2");
  if (failed(verifyTileBufLayoutConstraints(*this, resultTy, "result")))
    return failure();

  auto viewSemantics = (*this)->getAttrOfType<StringAttr>("pto.view_semantics");
  bool isSubview = viewSemantics && viewSemantics.getValue() == "subview";
  if (!isSubview && sourceTy.getShape() != resultTy.getShape())
    return emitOpError() << "source/result shape mismatch: source="
                         << sourceTy << " result=" << resultTy;

  if (sourceTy.getElementType() != resultTy.getElementType())
    return emitOpError() << "source/result element type mismatch: source="
                         << sourceTy.getElementType()
                         << " result=" << resultTy.getElementType();

  if (sourceTy.getMemorySpace() != resultTy.getMemorySpace())
    return emitOpError() << "source/result memory space mismatch";

  if (getConfig() != resultTy.getConfigAttr())
    return emitOpError("config attribute must match the result tile_buf config");

  auto shape = resultTy.getShape();
  auto validShape = resultTy.getValidShape();
  if (validShape.size() != 2)
    return emitOpError("result tile_buf must have rank-2 validShape");
  for (unsigned i = 0; i < 2; ++i) {
    if (shape[i] != ShapedType::kDynamic &&
        validShape[i] != ShapedType::kDynamic && validShape[i] > shape[i]) {
      return emitOpError() << "valid_shape[" << i << "] must be <= shape["
                           << i << "]";
    }
  }

  return success();
}

LogicalResult TAssignOp::verify() {
  if (getTile().getType() != getResult().getType()) {
    return emitOpError("result type must match tile operand type");
  }
  return success();
}

LogicalResult TLoadOp::verify() {
  auto verifyCommon =
      [&](bool allowLowPrecision)
      -> FailureOr<std::pair<pto::PartitionTensorViewType, pto::TileBufType>> {
    auto srcPart = dyn_cast<pto::PartitionTensorViewType>(getSrc().getType());
    auto dstTile = dyn_cast<pto::TileBufType>(getDst().getType());
    if (!srcPart || !dstTile) {
      emitOpError("expects src to be !pto.partition_tensor_view and dst to be !pto.tile_buf");
      return failure();
    }
    if (failed(verifyTileBufCommon(*this, dstTile, "dst", allowLowPrecision)))
      return failure();

    auto srcShape = srcPart.getShape();
    for (unsigned i = 0; i < srcShape.size(); ++i) {
      if (srcShape[i] != ShapedType::kDynamic && srcShape[i] <= 0) {
        emitOpError() << "expects src shape[" << i << "] to be positive";
        return failure();
      }
    }
    auto dstValid = dstTile.getValidShape();
    for (unsigned i = 0; i < dstValid.size(); ++i) {
      if (dstValid[i] != ShapedType::kDynamic && dstValid[i] < 0) {
        emitOpError() << "expects dst valid_shape[" << i << "] to be non-negative";
        return failure();
      }
    }
    return std::make_pair(srcPart, dstTile);
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/false);
    if (failed(common))
      return failure();
    auto [srcPart, dstTile] = *common;

    Type srcElem = srcPart.getElementType();
    Type dstElem = dstTile.getElementType();
    if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))
      return emitOpError("expects A2/A3 tload low-precision element types to be unsupported");
    if (!(dstElem.isInteger(8) || dstElem.isInteger(16) || dstElem.isInteger(32) ||
          dstElem.isInteger(64) || dstElem.isF16() || dstElem.isBF16() || dstElem.isF32()))
      return emitOpError("expects A2/A3 tload dst element type to be i8/i16/i32/i64/u64/f16/bf16/f32");

    auto dstSpace = getPTOMemorySpaceEnum(dstTile);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT))
      return emitOpError("expects A2/A3 tload dst to use loc=vec or loc=mat");

    if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
      return emitOpError("expects src and dst element types to have the same bitwidth");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/true);
    if (failed(common))
      return failure();
    auto [srcPart, dstTile] = *common;

    Type srcElem = srcPart.getElementType();
    Type dstElem = dstTile.getElementType();
    unsigned srcBytes = getElemByteSize(srcElem);
    unsigned dstBytes = getElemByteSize(dstElem);
    if (srcBytes != dstBytes)
      return emitOpError("expects src and dst element types to have the same element size");
    if (!(dstBytes == 1 || dstBytes == 2 || dstBytes == 4 || dstBytes == 8))
      return emitOpError("expects A5 tload dst element size to be 1, 2, 4, or 8 bytes");
    if (!isA5TLoadStoreTransferElemType(srcElem))
      return emitOpError("expects A5 tload src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
    if (!isA5TLoadStoreTransferElemType(dstElem))
      return emitOpError("expects A5 tload dst element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");

    if (dstElem.isInteger(64)) {
      auto pad = dstTile.getPadValueI32();
      if (pad != static_cast<int32_t>(pto::PadValue::Null) &&
          pad != static_cast<int32_t>(pto::PadValue::Zero))
        return emitOpError("expects A5 i64/u64 tload dst pad to be null or zero");
    }

    auto dstSpace = getPTOMemorySpaceEnum(dstTile);
    if (dstSpace && *dstSpace == pto::AddressSpace::VEC) {
      int32_t bl = dstTile.getBLayoutValueI32();
      int32_t sl = dstTile.getSLayoutValueI32();
      bool isND = (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isDN = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isNZ = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::RowMajor));
      if (!isND && !isDN && !isNZ)
        return emitOpError("expects A5 tload vec dst layout to be ND, DN, or NZ");
    }

    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TPrefetchOp::verify() {
  auto verifyImpl = [&](bool allowLowPrecision) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();

    Type srcElem;
    Type dstElem;

    if (auto srcPart = dyn_cast<pto::PartitionTensorViewType>(srcTy)) {
      auto srcShape = srcPart.getShape();
      for (unsigned i = 0; i < srcShape.size(); ++i) {
        if (srcShape[i] != ShapedType::kDynamic && srcShape[i] <= 0)
          return emitOpError() << "expects src shape[" << i << "] to be positive";
      }
      srcElem = srcPart.getElementType();
    } else if (auto srcMr = dyn_cast<MemRefType>(srcTy)) {
      if (!srcMr.hasRank())
        return emitOpError("expects src memref to be ranked");
      for (int64_t dim : srcMr.getShape()) {
        if (dim != ShapedType::kDynamic && dim <= 0)
          return emitOpError("expects src memref shape to be positive");
      }
      srcElem = srcMr.getElementType();
    } else {
      return emitOpError("expects src to be !pto.partition_tensor_view or memref");
    }

    if (auto dstTile = dyn_cast<pto::TileBufType>(dstTy)) {
      if (failed(verifyTileBufCommon(*this, dstTile, "dst", allowLowPrecision)))
        return failure();
      auto dstValid = dstTile.getValidShape();
      for (unsigned i = 0; i < dstValid.size(); ++i) {
        if (dstValid[i] != ShapedType::kDynamic && dstValid[i] < 0)
          return emitOpError() << "expects dst valid_shape[" << i
                               << "] to be non-negative";
      }
      auto dstSpace = getPTOMemorySpaceEnum(dstTile);
      if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                        *dstSpace != pto::AddressSpace::MAT))
        return emitOpError("expects dst to use loc=vec or loc=mat");
      dstElem = dstTile.getElementType();
    } else if (auto dstMr = dyn_cast<MemRefType>(dstTy)) {
      auto dstSpace = getPTOMemorySpaceEnum(dstMr);
      if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                        *dstSpace != pto::AddressSpace::MAT))
        return emitOpError("expects dst memref to use loc=vec or loc=mat");
      if (!dstMr.hasRank())
        return emitOpError("expects dst memref to be ranked");
      if (failed(verifyTileBufCommon(*this, dstMr, "dst", allowLowPrecision)))
        return failure();
      dstElem = dstMr.getElementType();
    } else {
      return emitOpError("expects dst to be !pto.tile_buf or memref");
    }

    if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
      return emitOpError("expects src and dst element types to have the same element size");
    if (!allowLowPrecision &&
        (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem)))
      return emitOpError("expects A2/A3 tprefetch low-precision element types to be unsupported");
    if (allowLowPrecision &&
        (!isA5TLoadStoreTransferElemType(srcElem) ||
         !isA5TLoadStoreTransferElemType(dstElem)))
      return emitOpError("expects A5 tprefetch element types to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyImpl(/*allowLowPrecision=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyImpl(/*allowLowPrecision=*/true);
  };
  switch (getVerifierTargetArch(getOperation())) {
  case VerifierTargetArch::A2A3:
    return verifyA2A3();
  case VerifierTargetArch::A5:
    return verifyA5();
  }
  return failure();
}

LogicalResult MakePrefetchAsyncContextOp::verify() {
  Type workspaceTy = getWorkspace().getType();
  Type elemTy = nullptr;
  if (auto ptrTy = dyn_cast<pto::PtrType>(workspaceTy)) {
    elemTy = ptrTy.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(workspaceTy)) {
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError("expects workspace memref to be in GM address space");
    elemTy = memTy.getElementType();
  } else {
    return emitOpError("expects workspace to be !pto.ptr<i8> or GM memref<i8>");
  }
  if (!isByteIntegerType(elemTy))
    return emitOpError("expects workspace element type to be an 8-bit integer");
  return success();
}

LogicalResult TPrefetchAsyncOp::verify() {
  if (failed(verifyAsyncFlatContiguous1DGMViewLike(getOperation(), getSrc(),
                                                   "src")))
    return failure();
  return success();
}

LogicalResult mlir::pto::SetFFTsOp::verify() {
  auto mr = llvm::dyn_cast<mlir::MemRefType>(getFfts().getType());
  if (!mr)
    return emitOpError("expects a memref operand");

  if (!mr.getElementType().isInteger(64) && !mr.getElementType().isInteger(8))
    return emitOpError("expects element type i64 (or i8)");

  return mlir::success();
}

ParseResult mlir::pto::SyncSetOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SyncSetOp::getPipeAttrName(result.name),
                                SyncSetOp::getEventIdAttrName(result.name));
}

void mlir::pto::SyncSetOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::SyncSetOp::verify() {
  bool hasStatic = getEventIdAttr() != nullptr;
  bool hasDynamic = static_cast<bool>(getEventIdDyn());
  if (hasStatic == hasDynamic)
    return emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";
  if (IntegerAttr fftsModeAttr = getFftsModeAttr()) {
    int64_t fftsMode = fftsModeAttr.getInt();
    if (fftsMode < 0 || fftsMode > 2)
      return emitOpError() << "requires ffts_mode in range [0, 2], but got "
                           << fftsMode;
  }

  auto verifyA2A3 = [&]() -> LogicalResult { return success(); };
  auto verifyA5 = [&]() -> LogicalResult {
    switch (getPipe().getPipe()) {
    case PIPE::PIPE_FIX:
    case PIPE::PIPE_MTE3:
      return success();
    default:
      return emitOpError()
             << "A5 sync.set expects pipe to be one of <PIPE_FIX>, <PIPE_MTE3>";
    }
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::SyncWaitOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SyncWaitOp::getPipeAttrName(result.name),
                                SyncWaitOp::getEventIdAttrName(result.name));
}

void mlir::pto::SyncWaitOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

ParseResult mlir::pto::SyncAllOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<Type, 4> operandTypes;
  Attribute modeAttr;
  Attribute coreTypeAttr;

  if (parser.parseLParen())
    return failure();

  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseOperandList(operands) || parser.parseColonTypeList(operandTypes) ||
        parser.parseRParen())
      return failure();
    if (operands.size() != operandTypes.size())
      return parser.emitError(parser.getCurrentLocation())
             << "expects the same number of operands and operand types";
  }

  if (parser.parseKeyword("mode") || parser.parseEqual() ||
      parser.parseAttribute(modeAttr) || parser.parseComma() ||
      parser.parseKeyword("core_type") || parser.parseEqual() ||
      parser.parseAttribute(coreTypeAttr))
    return failure();

  auto mode = dyn_cast<pto::SyncAllModeAttr>(modeAttr);
  if (!mode)
    return parser.emitError(parser.getCurrentLocation())
           << "expects mode to be #pto.sync_all_mode<...>";

  auto coreType = dyn_cast<pto::SyncCoreTypeAttr>(coreTypeAttr);
  if (!coreType)
    return parser.emitError(parser.getCurrentLocation())
           << "expects core_type to be #pto.sync_core_type<...>";

  result.addAttribute("mode", mode);
  result.addAttribute("core_type", coreType);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  auto addSegmentSizes = [&](int32_t gm, int32_t ub, int32_t l1,
                             int32_t used) {
    result.addAttribute("operandSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr(
                            {gm, ub, l1, used}));
  };

  switch (mode.getValue()) {
  case pto::SyncAllMode::Hard:
    if (!operands.empty())
      return parser.emitError(parser.getCurrentLocation())
             << "expects hard syncall to have no operands";
    addSegmentSizes(0, 0, 0, 0);
    return success();
  case pto::SyncAllMode::Soft:
    break;
  }

  switch (coreType.getValue()) {
  case pto::SyncCoreType::AIVOnly:
    if (operands.size() != 2 && operands.size() != 3)
      return parser.emitError(parser.getCurrentLocation())
             << "expects soft AIV-only syncall to have gm_workspace, "
                "ub_workspace, and optional used_cores";
    if (parser.resolveOperand(operands[0], operandTypes[0], result.operands) ||
        parser.resolveOperand(operands[1], operandTypes[1], result.operands))
      return failure();
    if (operands.size() == 3 &&
        parser.resolveOperand(operands[2], operandTypes[2], result.operands))
      return failure();
    addSegmentSizes(1, 1, 0, operands.size() == 3 ? 1 : 0);
    return success();
  case pto::SyncCoreType::AICOnly:
    if (operands.size() != 2 && operands.size() != 3)
      return parser.emitError(parser.getCurrentLocation())
             << "expects soft AIC-only syncall to have gm_workspace, "
                "l1_workspace, and optional used_cores";
    if (parser.resolveOperand(operands[0], operandTypes[0], result.operands) ||
        parser.resolveOperand(operands[1], operandTypes[1], result.operands))
      return failure();
    if (operands.size() == 3 &&
        parser.resolveOperand(operands[2], operandTypes[2], result.operands))
      return failure();
    addSegmentSizes(1, 0, 1, operands.size() == 3 ? 1 : 0);
    return success();
  case pto::SyncCoreType::Mix:
    if (operands.size() != 3 && operands.size() != 4)
      return parser.emitError(parser.getCurrentLocation())
             << "expects soft mixed syncall to have gm_workspace, "
                "ub_workspace, l1_workspace, and optional used_cores";
    if (parser.resolveOperand(operands[0], operandTypes[0], result.operands) ||
        parser.resolveOperand(operands[1], operandTypes[1], result.operands) ||
        parser.resolveOperand(operands[2], operandTypes[2], result.operands))
      return failure();
    if (operands.size() == 4 &&
        parser.resolveOperand(operands[3], operandTypes[3], result.operands))
      return failure();
    addSegmentSizes(1, 1, 1, operands.size() == 4 ? 1 : 0);
    return success();
  }

  llvm_unreachable("unhandled SyncCoreType");
}

void mlir::pto::SyncAllOp::print(OpAsmPrinter &p) {
  SmallVector<Value, 4> operands;
  if (getGmWorkspace())
    operands.push_back(getGmWorkspace());
  if (getUbWorkspace())
    operands.push_back(getUbWorkspace());
  if (getL1Workspace())
    operands.push_back(getL1Workspace());
  if (getUsedCores())
    operands.push_back(getUsedCores());

  p << "(";
  if (!operands.empty()) {
    p.printOperands(operands);
    p << " : ";
    llvm::interleaveComma(operands, p,
                          [&](Value operand) { p.printType(operand.getType()); });
  }
  p << ") mode = " << getMode() << ", core_type = " << getCoreType();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes", "mode",
                                           "core_type"});
}

LogicalResult mlir::pto::SyncWaitOp::verify() {
  bool hasStatic = getEventIdAttr() != nullptr;
  bool hasDynamic = static_cast<bool>(getEventIdDyn());
  if (hasStatic == hasDynamic)
    return emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";

  auto verifyA2A3 = [&]() -> LogicalResult { return success(); };
  auto verifyA5 = [&]() -> LogicalResult {
    switch (getPipe().getPipe()) {
    case PIPE::PIPE_FIX:
    case PIPE::PIPE_MTE1:
    case PIPE::PIPE_MTE2:
    case PIPE::PIPE_MTE3:
    case PIPE::PIPE_V:
      return success();
    default:
      return emitOpError() << "A5 sync.wait expects pipe to be one of "
                              "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, "
                              "<PIPE_MTE3>, <PIPE_V>";
    }
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TStoreOp::verify() {
  auto verifyCommon =
      [&](bool allowLowPrecision)
      -> FailureOr<std::pair<pto::TileBufType, pto::PartitionTensorViewType>> {
    auto srcTile = dyn_cast<pto::TileBufType>(getSrc().getType());
    auto dstPart = dyn_cast<pto::PartitionTensorViewType>(getDst().getType());
    if (!srcTile || !dstPart) {
      emitOpError("expects src to be !pto.tile_buf and dst to be !pto.partition_tensor_view");
      return failure();
    }
    if (failed(verifyTileBufCommon(*this, srcTile, "src", allowLowPrecision)))
      return failure();
    for (auto [idx, dim] : llvm::enumerate(dstPart.getShape())) {
      if (dim != ShapedType::kDynamic && dim <= 0) {
        emitOpError() << "expects dst shape[" << idx << "] to be positive";
        return failure();
      }
    }
    auto srcValid = srcTile.getValidShape();
    for (auto [idx, dim] : llvm::enumerate(srcValid)) {
      if (dim != ShapedType::kDynamic && dim < 0) {
        emitOpError() << "expects src valid_shape[" << idx << "] to be non-negative";
        return failure();
      }
    }

    // Keep TSTORE contract explicit while preserving existing legal layout
    // reinterpretation paths (e.g. 1x1024 <-> 32x32, 5D partition views).
    // When both sides are fully static, require equal element counts between
    // dst shape and src valid_shape.
    auto getStaticElemCount = [](ArrayRef<int64_t> shape) -> std::optional<int64_t> {
      int64_t total = 1;
      for (int64_t dim : shape) {
        if (dim == ShapedType::kDynamic)
          return std::nullopt;
        if (dim <= 0)
          return std::nullopt;
        if (total > std::numeric_limits<int64_t>::max() / dim)
          return std::nullopt;
        total *= dim;
      }
      return total;
    };

    auto dstElemCount = getStaticElemCount(dstPart.getShape());
    auto srcValidElemCount = getStaticElemCount(srcValid);
    if (dstElemCount && srcValidElemCount && *dstElemCount != *srcValidElemCount) {
      emitOpError() << "expects dst static element count (" << *dstElemCount
                    << ") to match src valid_shape static element count ("
                    << *srcValidElemCount << ")";
      return failure();
    }
    return std::make_pair(srcTile, dstPart);
  };

  auto isLoadStoreElemType = [&](Type ty) -> bool {
    return ty.isInteger(8) || ty.isInteger(16) || ty.isInteger(32) ||
           ty.isInteger(64) || ty.isF16() || ty.isBF16() || ty.isF32();
  };
  auto isI8Like = [&](Type ty) -> bool { return ty.isInteger(8); };
  bool hasPreQuant = static_cast<bool>(getPreQuantScalar());
  auto reluMode = getReluPreMode();

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/false);
    if (failed(common))
      return failure();
    auto [srcTile, dstPart] = *common;
    auto srcSpace = getPTOMemorySpaceEnum(srcTile);
    if (!srcSpace || (*srcSpace != pto::AddressSpace::VEC &&
                      *srcSpace != pto::AddressSpace::MAT &&
                      *srcSpace != pto::AddressSpace::ACC))
      return emitOpError("expects A2/A3 tstore src to use loc=vec, loc=mat, or loc=acc");
    if (hasPreQuant && *srcSpace != pto::AddressSpace::ACC)
      return emitOpError("expects preQuantScalar form to use loc=acc src");
    if (reluMode != pto::ReluPreMode::NoRelu && *srcSpace != pto::AddressSpace::ACC)
      return emitOpError("expects reluPreMode form to use loc=acc src");

    Type srcElem = srcTile.getElementType();
    Type dstElem = dstPart.getElementType();
    if (*srcSpace == pto::AddressSpace::VEC || *srcSpace == pto::AddressSpace::MAT) {
      if (hasPreQuant)
        return emitOpError("expects preQuantScalar form to use loc=acc src");
      if (isPTOLowPrecisionType(dstElem))
        return emitOpError("expects A2/A3 vec/mat tstore low-precision dst element types to be unsupported");
      if (!isLoadStoreElemType(srcElem))
        return emitOpError("expects A2/A3 vec/mat tstore src element type to be i8/i16/i32/i64/u64/f16/bf16/f32");
      if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
        return emitOpError("expects A2/A3 vec/mat tstore src and dst element types to have the same bitwidth");
      return success();
    }

    if (!(srcElem.isInteger(32) || srcElem.isF32()))
      return emitOpError("expects A2/A3 acc tstore src element type to be i32 or f32");
    if (hasPreQuant) {
      if (srcElem.isInteger(32)) {
        if (!(isI8Like(dstElem) || dstElem.isF16()))
          return emitOpError("expects A2/A3 acc preQuantScalar tstore dst type to be i8/ui8/f16");
      } else if (srcElem.isF32()) {
        if (!isI8Like(dstElem))
          return emitOpError("expects A2/A3 acc preQuantScalar tstore dst type to be i8/ui8");
      }
    } else {
      if (!(dstElem.isInteger(32) || dstElem.isF32() || dstElem.isF16() ||
            dstElem.isBF16()))
        return emitOpError("expects A2/A3 acc tstore dst element type to be i32/f32/f16/bf16");
    }

    auto srcShape = srcTile.getShape();
    if (srcShape[1] != ShapedType::kDynamic &&
        (srcShape[1] < 1 || srcShape[1] > 4095))
      return emitOpError("expects A2/A3 acc tstore src cols to be in [1, 4095]");
    auto srcValid = srcTile.getValidShape();
    if (srcValid[1] != ShapedType::kDynamic &&
        (srcValid[1] < 0 || srcValid[1] > 4095))
      return emitOpError("expects A2/A3 acc tstore src valid_shape[1] to be in [0, 4095]");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/true);
    if (failed(common))
      return failure();
    auto [srcTile, dstPart] = *common;
    auto srcSpace = getPTOMemorySpaceEnum(srcTile);
    if (!srcSpace || (*srcSpace != pto::AddressSpace::VEC &&
                      *srcSpace != pto::AddressSpace::ACC))
      return emitOpError("expects A5 tstore src to use loc=vec or loc=acc");
    if (hasPreQuant && *srcSpace != pto::AddressSpace::ACC)
      return emitOpError("expects preQuantScalar form to use loc=acc src");
    if (reluMode != pto::ReluPreMode::NoRelu && *srcSpace != pto::AddressSpace::ACC)
      return emitOpError("expects reluPreMode form to use loc=acc src");

    Type srcElem = srcTile.getElementType();
    Type dstElem = dstPart.getElementType();
    if (*srcSpace == pto::AddressSpace::VEC) {
      if (hasPreQuant)
        return emitOpError("expects preQuantScalar form to use loc=acc src");
      if (!isA5TLoadStoreTransferElemType(srcElem))
        return emitOpError("expects A5 vec tstore src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
      if (getElemByteSize(srcElem) != getElemByteSize(dstElem))
        return emitOpError("expects A5 vec tstore src and dst element types to have the same bitwidth");

      int32_t bl = srcTile.getBLayoutValueI32();
      int32_t sl = srcTile.getSLayoutValueI32();
      bool isND = (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isDN = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isNZ = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::RowMajor));
      auto srcShape = srcTile.getShape();
      bool isSpecialCase = (srcShape.size() == 2 && (srcShape[0] == 1 || srcShape[1] == 1));
      if (!isSpecialCase && !isND && !isDN && !isNZ)
        return emitOpError("expects A5 vec tstore src layout to be ND, DN, or NZ (or special case with 1 row/col)");
      return success();
    }

    if (!(srcElem.isInteger(32) || srcElem.isF32()))
      return emitOpError("expects A5 acc tstore src element type to be i32 or f32");
    if (hasPreQuant) {
      if (!isA5AccStorePreQuantDstType(srcElem, dstElem))
        return emitOpError("expects A5 acc preQuantScalar tstore dst type to be i8/ui8/f16/bf16/f32/hif8/f8E4M3");
    } else {
      if (!(dstElem.isInteger(32) || dstElem.isF32() || dstElem.isF16() ||
            dstElem.isBF16()))
        return emitOpError("expects A5 acc tstore dst element type to be i32/f32/f16/bf16");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAbsOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();

  Type elemTy;
  if (auto tb = dyn_cast<pto::TileBufType>(srcTy))
    elemTy = tb.getElementType();
  else if (auto mr = dyn_cast<MemRefType>(srcTy))
    elemTy = mr.getElementType();
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";

  return success();
}
// PTO.cpp

static bool isPTOShapedLike(Type ty) {
  return mlir::isa<MemRefType, RankedTensorType,
                pto::TensorViewType, pto::TileBufType,
                pto::PartitionTensorViewType>(ty);
}

static bool isTileLikeType(Type ty) {
  return isa<pto::TileBufType, MemRefType>(ty);
}

static Type getElemTy(Type ty) {
  if (auto mr = mlir::dyn_cast<MemRefType>(ty)) return mr.getElementType();
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty)) return tt.getElementType();
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty)) return tv.getElementType();
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty)) return tb.getElementType();
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty)) return tv.getElementType();
  return Type();
}

static SmallVector<int64_t, 4> getShapeVec(Type ty) {
  SmallVector<int64_t, 4> s;
  if (auto mr = mlir::dyn_cast<MemRefType>(ty))
    return SmallVector<int64_t,4>(mr.getShape().begin(), mr.getShape().end());
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty))
    return SmallVector<int64_t,4>(tt.getShape().begin(), tt.getShape().end());
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty))
    return SmallVector<int64_t,4>(tv.getShape().begin(), tv.getShape().end());
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty))
    return SmallVector<int64_t,4>(tb.getShape().begin(), tb.getShape().end());
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty))
    return SmallVector<int64_t,4>(tv.getShape().begin(), tv.getShape().end());
  return {};
}

static SmallVector<int64_t, 4> getValidShapeVec(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty))
    return SmallVector<int64_t, 4>(tb.getValidShape().begin(), tb.getValidShape().end());
  return getShapeVec(ty);
}

static int64_t getLogicalTileDim(int64_t rawDim, Type elemTy,
                                 std::optional<pto::BLayout> blayout,
                                 unsigned dimIdx) {
  if (rawDim == ShapedType::kDynamic || !isPTOFloat4PackedType(elemTy))
    return rawDim;
  pto::BLayout layout = blayout.value_or(pto::BLayout::RowMajor);
  unsigned packedDim = layout == pto::BLayout::ColMajor ? 0 : 1;
  return dimIdx == packedDim ? rawDim * 2 : rawDim;
}

static std::optional<pto::BLayout> getTileBufBLayout(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty))
    return static_cast<pto::BLayout>(tb.getBLayoutValueI32());
  return std::nullopt;
}

static SmallVector<int64_t, 4> getLogicalTileExtentVec(Type ty,
                                                       bool useValidShape) {
  SmallVector<int64_t, 4> dims =
      useValidShape ? getValidShapeVec(ty) : getShapeVec(ty);
  if (!isTileLikeType(ty) || dims.size() != 2)
    return dims;

  Type elemTy = getElemTy(ty);
  auto blayout = getTileBufBLayout(ty);
  for (unsigned i = 0; i < dims.size(); ++i)
    dims[i] = getLogicalTileDim(dims[i], elemTy, blayout, i);
  return dims;
}

static int64_t getConstantIndexOrDynamic(Value value) {
  if (!value)
    return ShapedType::kDynamic;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return ShapedType::kDynamic;
}

static SmallVector<int64_t, 4> getValidShapeVec(Value value) {
  if (!value)
    return {};
  auto valid = getValidShapeVec(value.getType());
  if (auto bind = value.getDefiningOp<pto::BindTileOp>()) {
    if (valid.size() >= 1 && bind.getValidRow())
      valid[0] = getConstantIndexOrDynamic(bind.getValidRow());
    if (valid.size() >= 2 && bind.getValidCol())
      valid[1] = getConstantIndexOrDynamic(bind.getValidCol());
  }
  return valid;
}

static SmallVector<int64_t, 4> getMatmulLogicalShapeVec(Type ty) {
  auto shape = getShapeVec(ty);
  auto valid = getValidShapeVec(ty);
  if (!isa<pto::TileBufType>(ty) || shape.size() != valid.size())
    return shape;

  for (size_t i = 0, e = shape.size(); i < e; ++i) {
    if (valid[i] != ShapedType::kDynamic)
      shape[i] = valid[i];
  }
  return shape;
}

static bool isByteIntegerType(Type ty) {
  auto intTy = dyn_cast<IntegerType>(ty);
  return intTy && intTy.getWidth() == 8;
}

static LogicalResult verifyAsyncFlatContiguous1DGMMemRef(Operation *op,
                                                         Value value,
                                                         StringRef name) {
  auto memTy = dyn_cast<MemRefType>(value.getType());
  if (!memTy)
    return op->emitOpError() << "expects " << name << " to be a memref";
  if (!memTy.hasRank())
    return op->emitOpError() << "expects " << name << " to be a ranked memref";
  if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
    return op->emitOpError() << "expects " << name
                             << " to be in GM address space";

  ArrayRef<int64_t> shape = memTy.getShape();
  if (shape.empty())
    return op->emitOpError() << "expects " << name
                             << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic)
      return op->emitOpError() << "expects " << name
                               << " to have a static shape";
  }

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(getStridesAndOffset(memTy, strides, offset)))
    return op->emitOpError() << "expects " << name
                             << " to be a strided memref with a known layout";

  bool hasDynamicLayout =
      offset == ShapedType::kDynamic ||
      llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      });
  if (hasDynamicLayout)
    return success();

  bool packed = !strides.empty() && strides.back() == 1;
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0 && packed; --i)
    packed &= strides[i] == strides[i + 1] * shape[i + 1];
  if (!packed)
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM memref";

  bool logical1D = true;
  for (int i = 0, e = static_cast<int>(shape.size()) - 1; i < e; ++i)
    logical1D &= shape[i] == 1;
  if (!logical1D)
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM memref";

  return success();
}

static LogicalResult verifyAsyncFlatContiguous1DGMViewLike(Operation *op,
                                                           Value value,
                                                           StringRef name) {
  Type ty = value.getType();
  if (isa<MemRefType>(ty))
    return verifyAsyncFlatContiguous1DGMMemRef(op, value, name);

  if (!isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a memref/tensor_view/partition_view";

  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic)
      return op->emitOpError() << "expects " << name
                               << " to have a static shape";
  }

  bool logical1D = true;
  for (int i = 0, e = static_cast<int>(shape.size()) - 1; i < e; ++i)
    logical1D &= shape[i] == 1;
  if (!logical1D)
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM view";

  return success();
}

static bool isCommGlobalLikeType(Type ty) {
  if (auto memTy = dyn_cast<MemRefType>(ty))
    return isGmAddressSpaceAttr(memTy.getMemorySpace());
  return isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty);
}

static LogicalResult verifyCommGlobalLike(Operation *op, Value value,
                                          StringRef name) {
  Type ty = value.getType();
  if (!isCommGlobalLikeType(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a GM memref/tensor_view/partition_view";

  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " to have a positive static shape";
  }
  return success();
}

static LogicalResult verifyCommSignalLike(Operation *op, Value value,
                                          StringRef name) {
  if (failed(verifyCommGlobalLike(op, value, name)))
    return failure();
  Type elemTy = getElemTy(value.getType());
  if (!elemTy || !elemTy.isSignlessInteger(32))
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";
  return success();
}

static LogicalResult verifyCommStagingTileLike(Operation *op, Value value,
                                               StringRef name) {
  Type ty = value.getType();
  if (!isa<pto::TileBufType, MemRefType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a tile_buf or memref tile";
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC)
    return op->emitOpError() << "expects " << name
                             << " to be in vec address space";
  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " to have a positive static shape";
  }
  return success();
}

static LogicalResult verifyCommGlobalGroup(Operation *op, ValueRange group,
                                           StringRef name) {
  if (group.empty())
    return op->emitOpError() << "expects at least one " << name << " operand";
  Type groupTy = group.front().getType();
  for (auto it : llvm::enumerate(group)) {
    if (failed(verifyCommGlobalLike(op, it.value(),
                                    (name + "[" + Twine(it.index()) + "]").str())))
      return failure();
    if (it.value().getType() != groupTy)
      return op->emitOpError() << "expects all " << name
                               << " operands to have identical types";
  }
  return success();
}

static LogicalResult verifyCommPingPongSameType(Operation *op, Value ping,
                                                Value pong, StringRef pingName,
                                                StringRef pongName) {
  if (!pong)
    return success();
  if (failed(verifyCommStagingTileLike(op, ping, pingName)) ||
      failed(verifyCommStagingTileLike(op, pong, pongName)))
    return failure();
  if (ping.getType() != pong.getType())
    return op->emitOpError() << "expects " << pingName << " and " << pongName
                             << " to have identical types";
  return success();
}

static std::optional<uint64_t> getStaticByteSize(Type ty) {
  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty())
    return std::nullopt;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim < 0)
      return std::nullopt;
  }

  Type elemTy = getElemTy(ty);
  uint64_t elemBytes = getElemByteSize(elemTy);
  if (elemBytes == 0)
    return std::nullopt;

  uint64_t total = elemBytes;
  for (int64_t dim : shape) {
    total *= static_cast<uint64_t>(dim);
  }
  return total;
}

static std::optional<pto::AddressSpace> getPTOMemorySpaceEnum(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(tb.getMemorySpace()))
      return as.getAddressSpace();
    return std::nullopt;
  }
  if (auto mr = dyn_cast<MemRefType>(ty)) {
    if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(mr.getMemorySpace()))
      return as.getAddressSpace();
    if (!mr.getMemorySpace())
      return pto::AddressSpace::GM;
  }
  return std::nullopt;
}

[[maybe_unused]] static bool isRank2TileBuf(Type ty) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getRank() == 2 && tb.getValidShape().size() == 2;
}

static bool isSupportedVecElemType(Type ty, bool allowBf16,
                                   bool allowInt8) {
  if (ty.isF16() || ty.isF32())
    return true;
  if (allowBf16 && ty.isBF16())
    return true;
  if (auto it = dyn_cast<IntegerType>(ty)) {
    switch (it.getWidth()) {
    case 32:
    case 16:
      return true;
    case 8:
      return allowInt8;
    default:
      return false;
    }
  }
  return false;
}

static bool isSupportedMGatherMScatterIndexElemType(Type ty) {
  auto it = dyn_cast<IntegerType>(ty);
  if (!it || it.getWidth() != 32)
    return false;
  return true;
}

static bool isSupportedMGatherMScatterPayloadElemType(Operation *op, Type ty) {
  if (isSupportedVecElemType(ty, /*allowBf16=*/true, /*allowInt8=*/true))
    return true;
  if (!isTargetArchA5(op))
    return false;
  return ty.isFloat8E4M3() || ty.isFloat8E4M3FN() || ty.isFloat8E4M3FNUZ() ||
         ty.isFloat8E4M3B11FNUZ() || ty.isFloat8E5M2() || ty.isFloat8E5M2FNUZ();
}

static bool isSupportedMScatterAtomicPayloadElemType(Type ty,
                                                     pto::ScatterAtomicOp atomic) {
  auto intTy = dyn_cast<IntegerType>(ty);
  switch (atomic) {
  case pto::ScatterAtomicOp::None:
    return true;
  case pto::ScatterAtomicOp::Add:
    return ty.isF16() || ty.isF32() ||
           (intTy && intTy.getWidth() == 32);
  case pto::ScatterAtomicOp::Max:
  case pto::ScatterAtomicOp::Min:
    return ty.isF32() ||
           (intTy && intTy.getWidth() == 32);
  }
  llvm_unreachable("Unknown ScatterAtomicOp");
}

static LogicalResult verifyMGatherMScatterMemOperand(Operation *op,
                                                     Value memValue,
                                                     Type dataElemTy,
                                                     StringRef dataOperandLabel) {
  Type memTy = memValue.getType();
  Type memElem = getElemTy(memTy);
  if (!memElem || memElem != dataElemTy)
    return op->emitOpError() << "expects mem element type to match "
                             << dataOperandLabel << " element type";

  if (isa<pto::PartitionTensorViewType>(memTy)) {
    if (auto layout = getLogicalViewLayout(memValue)) {
      if (*layout != pto::Layout::ND)
        return op->emitOpError(
            "expects mem partition view to use ND logical layout when layout "
            "can be inferred");
    }
    return success();
  }

  if (auto mr = dyn_cast<MemRefType>(memTy)) {
    auto as = getPTOMemorySpaceEnum(mr);
    if (!as || (*as != pto::AddressSpace::GM &&
                 *as != pto::AddressSpace::Zero))
      return op->emitOpError(
          "expects mem memref to use GM or zero address space");
    if (mr.getRank() == 5) {
      auto shape = mr.getShape();
      bool allStatic = true;
      for (int64_t d : shape)
        if (d == ShapedType::kDynamic)
          allStatic = false;
      if (allStatic && (shape[0] != 1 || shape[1] != 1 || shape[2] != 1))
        return op->emitOpError(
            "expects rank-5 GM memref leading dimensions to be [1,1,1,...] "
            "(GlobalTensor table shape)");
    }
    return success();
  }

  return op->emitOpError(
      "expects mem to be !pto.partition_tensor_view or a GM/ZERO memref");
}

static bool hasCompatibleKnownExtent(int64_t lhs, int64_t rhs);
static bool isKnownUnitExtent(int64_t value);
static bool isKnownZeroOrUnitExtent(int64_t value);
static bool hasCompatibleKnownExtentOrZero(int64_t lhs, int64_t rhs);

static LogicalResult verifyMGatherMScatterTileShape(Operation *op, Type dataTy,
                                                    Type idxTy,
                                                    StringRef dataName) {
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
  const bool elemCoalesce =
      hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      hasCompatibleKnownExtent(idxValid[1], dataValid[1]);

  if (!(rowCoalesce1xR || rowCoalesceRx1 || elemCoalesce))
    return op->emitOpError()
           << "expects idx valid_shape to be [0|1, " << dataName
           << ".valid_row], [" << dataName
           << ".valid_row, 0|1], or match " << dataName << " valid_shape";

  return success();
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

static LogicalResult verifyScaleTileMatchesOperand(Operation *op, Type scaleTy,
                                                   Type operandTy,
                                                   StringRef scaleName,
                                                   StringRef operandName) {
  if (failed(verifyTileBufCommon(op, scaleTy, scaleName)))
    return failure();
  auto scaleSpace = getPTOMemorySpaceEnum(scaleTy);
  if (!scaleSpace || *scaleSpace != pto::AddressSpace::SCALING)
    return op->emitOpError() << "expects " << scaleName
                             << " to be in the scaling address space";

  auto scaleShape = getShapeVec(scaleTy);
  auto operandShape = getShapeVec(operandTy);
  if (scaleShape.size() != operandShape.size())
    return op->emitOpError() << "expects " << scaleName << " and " << operandName
                             << " to have the same rank";
  for (size_t i = 0; i < scaleShape.size(); ++i) {
    if (scaleShape[i] != ShapedType::kDynamic &&
        operandShape[i] != ShapedType::kDynamic &&
        scaleShape[i] != operandShape[i])
      return op->emitOpError() << "expects " << scaleName << " and " << operandName
                               << " to have the same shape";
  }

  auto scaleValid = getValidShapeVec(scaleTy);
  auto operandValid = getValidShapeVec(operandTy);
  if (scaleValid.size() != operandValid.size())
    return op->emitOpError() << "expects " << scaleName << " and " << operandName
                             << " to have the same valid_shape";
  for (size_t i = 0; i < scaleValid.size(); ++i) {
    if (scaleValid[i] != ShapedType::kDynamic &&
        operandValid[i] != ShapedType::kDynamic &&
        scaleValid[i] != operandValid[i])
      return op->emitOpError() << "expects " << scaleName << " and " << operandName
                               << " to have the same valid_shape";
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

static LogicalResult verifyTColArgReductionOpCommon(Operation *op, Type srcTy,
                                                    Type tmpTy, Type dstTy) {
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyColArgReductionDstLayout(op, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp")) ||
      failed(verifyTileBufSameValidShape(op, srcTy, tmpTy, "src", "tmp")))
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
                                               Type rhsTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs")) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs")) ||
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
                                             Type rhsTy, Type dstTy) {
  if (failed(verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy)))
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
                                           Type dstTy) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy);
  case VerifierTargetArch::A5:
    return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy);
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

  if (isA5 && dstElemTy.isF32() && lhsElemTy == rhsElemTy) {
    if (auto ft = mlir::dyn_cast<FloatType>(lhsElemTy)) {
      unsigned width = ft.getWidth();
      if (width == 8 || width == 16 || width == 32)
        return success();
    }
  }

  return op->emitOpError()
         << "expects (dst, lhs, rhs) element types to match one of "
            "(i32, i8, i8), (f32, f16, f16), (f32, bf16, bf16), (f32, f32, f32)"
            << (isA5 ? ", or an A5-supported fp8 pair" : "");
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

LogicalResult pto::TCIOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
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

LogicalResult pto::TCmpOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyVecTileStorage(*this, t0, "src0")) ||
        failed(verifyVecTileStorage(*this, t1, "src1")) ||
        failed(verifyVecTileStorage(*this, td, "dst")))
      return failure();

    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    Type ed = getElemTy(td);
    if (!e0 || !e1 || !ed)
      return emitOpError("failed to get element type for src0/src1/dst");
    if (e0 != e1)
      return emitOpError("expects src0 and src1 to have the same element type");
    if (!(e0.isInteger(32) || e0.isF16() || e0.isF32()))
      return emitOpError("expects A2/A3 tcmp input element type to be i32/f16/f32");
    if (!ed.isInteger(8))
      return emitOpError("expects dst element type to be i8");

    auto valid0 = getValidShapeVec(t0);
    auto valid1 = getValidShapeVec(t1);
    auto validd = getValidShapeVec(td);
    if (valid0.size() != 2 || valid1.size() != 2 || validd.size() != 2)
      return emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
    if (!hasCompatibleKnownExtent(valid0[0], valid1[0]))
      return emitOpError("expects src0 and src1 to have the same valid row");
    if (!hasCompatibleKnownExtent(valid0[1], valid1[1]))
      return emitOpError("expects src0 and src1 to have the same valid column");
    if (!hasCompatibleKnownExtent(valid0[0], validd[0]))
      return emitOpError("expects src0 valid row to equal dst valid row");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
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
    if (!e0 || !e1 || !ed)
      return emitOpError("failed to get element type for src0/src1/dst");
    if (e0 != e1)
      return emitOpError("expects src0 and src1 to have the same element type");
    bool inputOk = e0.isF16() || e0.isF32() || e0.isBF16() ||
                   e0.isInteger(8) || e0.isInteger(16) || e0.isInteger(32);
    if (!inputOk)
      return emitOpError("expects A5 tcmp input element type to be i8/i16/i32/f16/bf16/f32");
    if (auto it = dyn_cast<IntegerType>(ed)) {
      if (it.getWidth() != 8)
        return emitOpError("expects dst element type to be i8");
    } else {
      return emitOpError("expects dst element type to be i8");
    }

    if (getShapeVec(t0) != getShapeVec(t1) || getShapeVec(t0) != getShapeVec(td))
      return emitOpError("expects src0, src1, and dst to have the same shape");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- TCMPS verify ----
LogicalResult pto::TCmpSOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32()))
      return emitOpError("expects A2/A3 tcmps input element type to be i16/i32/f16/f32");

    auto scalarTy = getScalar().getType();
    if (!(scalarTy.isIntOrIndexOrFloat()))
      return emitOpError("expects scalar to be integer, index, or float");

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return emitOpError("expects src and dst to have rank-2 valid_shape");
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0])
      return emitOpError("expects src and dst to have the same valid_shape[0]");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32()))
      return emitOpError("expects A5 tcmps input element type to be i8/i16/i32/f16/f32");

    auto scalarTy = getScalar().getType();
    if (!(scalarTy.isIntOrIndexOrFloat()))
      return emitOpError("expects scalar to be integer, index, or float");

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return emitOpError("expects src and dst to have rank-2 valid_shape");
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0])
      return emitOpError("expects src and dst to have the same valid_shape[0]");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColExpandOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
    return failure();
  if (getElemTy(srcTy) != getElemTy(dstTy))
    return emitOpError("expects src and dst to have the same element type");
  if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                              /*allowInt8=*/true))
    return emitOpError("expects tcolexpand element type to be supported");
  auto srcValid = getValidShapeVec(getSrc());
  auto dstValid = getValidShapeVec(getDst());
  if (srcValid.size() != 2 || dstValid.size() != 2)
    return emitOpError("expects src and dst to have rank-2 valid_shape");
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1])
    return emitOpError("expects src and dst to have the same valid_shape[1]");
  return success();
}
static LogicalResult verifyTColExpandBinaryLikeOp(Operation *op, Type t0, Type t1,
                                                  Type td, PTOArch targetArch,
                                                  StringRef opName,
                                                  bool allowIntegerTypes) {
  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) || !isPTOShapedLike(td))
    return op->emitOpError("expects src0/src1/dst to be PTO shaped-like types");

  Type e0 = getElemTy(t0);
  Type e1 = getElemTy(t1);
  Type ed = getElemTy(td);
  if (!e0 || !e1 || !ed)
    return op->emitOpError("failed to get element type for src0/src1/dst");

  auto isSupportedElem = [&](Type elemTy) {
    if (elemTy.isF16() || elemTy.isF32())
      return true;
    if (!allowIntegerTypes)
      return false;
    if (elemTy.isInteger(16) || elemTy.isInteger(32))
      return true;
    return targetArch == PTOArch::A5 && elemTy.isInteger(8);
  };
  if (!isSupportedElem(e0) || !isSupportedElem(e1) || !isSupportedElem(ed)) {
    if (!allowIntegerTypes)
      return op->emitOpError() << "expects " << opName
                               << " element type to be f16 or f32";
    if (targetArch == PTOArch::A5)
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i8/i16/i32/f16/f32";
    return op->emitOpError() << "expects A2/A3 " << opName
                             << " element type to be i16/i32/f16/f32";
  }

  if (getShapeVec(t0) != getShapeVec(td))
    return op->emitOpError("expects src0/dst to have same shape");
  if (failed(verifyTileBufSameValidShape(op, t0, td, "src0", "dst")))
    return failure();

  if (auto src0TileTy = dyn_cast<TileBufType>(t0)) {
    if (src0TileTy.getBLayoutValueI32() != 0)
      return op->emitOpError("expects src0 to use row-major layout");
  }

  if (auto src1TileTy = dyn_cast<TileBufType>(t1)) {
    if (src1TileTy.getBLayoutValueI32() != 0)
      return op->emitOpError("expects src1 to use row-major layout");
  }
  if (auto dstTileTy = dyn_cast<TileBufType>(td)) {
    if (dstTileTy.getBLayoutValueI32() != 0)
      return op->emitOpError("expects dst to use row-major layout");
  }

  auto src1Valid = getValidShapeVec(t1);
  auto dstValid = getValidShapeVec(td);
  if (src1Valid.size() == 2 && dstValid.size() == 2 &&
      src1Valid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      src1Valid[1] != dstValid[1])
    return op->emitOpError("expects src1 valid_shape[1] to equal dst valid_shape[1]");

  return success();
}
LogicalResult pto::TColExpandMulOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmul",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandAddOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandadd",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandDivOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    bool allowIntegerTypes = (targetArch == PTOArch::A5);
    return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        targetArch, "tcolexpanddiv",
                                        /*allowIntegerTypes=*/allowIntegerTypes);
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColExpandSubOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandsub",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandExpdifOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandexpdif",
                                      /*allowIntegerTypes=*/false);
}
LogicalResult pto::TColExpandMaxOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmax",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandMinOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmin",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColMaxOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/true,
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolmax element type to be f16/f32/i16/i32",
      "expects A5 tcolmax element type to be i8/i16/i32/f16/bf16/f32");
}

LogicalResult pto::TColArgMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTColArgReductionOpCommon(*this, getSrc().getType(),
                                          getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

LogicalResult pto::TColMinOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/true,
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolmin element type to be f16/f32/i16/i32",
      "expects A5 tcolmin element type to be i8/i16/i32/f16/bf16/f32");
}

LogicalResult pto::TColArgMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTColArgReductionOpCommon(*this, getSrc().getType(),
                                          getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}



ParseResult mlir::pto::TColSumOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  OpAsmParser::UnresolvedOperand dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  // Parse: ins(%src : type) or ins(%src, %tmp {isBinary = ...}: type, type)
  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();

  // Check for optional tmp operand (format 2)
  if (succeeded(parser.parseOptionalComma())) {
    // Format 2: ins(%src, %tmp {isBinary = ...}: type, type)
    if (parser.parseOperand(tmp))
      return failure();
    hasTmp = true;

    // Parse attributes (isBinary)
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();

    // Parse types: : type, type
    if (parser.parseColonType(srcTy) || parser.parseComma() || parser.parseType(tmpTy))
      return failure();
  } else {
    // Format 1: ins(%src : type)
    if (parser.parseColonType(srcTy))
      return failure();
  }

  if (parser.parseRParen())
    return failure();

  // Parse: outs(%dst : type)
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();

  // Parse any remaining attributes (for format 1)
  if (!hasTmp) {
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();
  }

  // Resolve operands
  if (parser.resolveOperand(src, srcTy, result.operands))
    return failure();

  if (hasTmp) {
    if (parser.resolveOperand(tmp, tmpTy, result.operands))
      return failure();
  }

  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  return success();
}

void mlir::pto::TColSumOp::print(OpAsmPrinter &p) {
  if (getTmp()) {
    // Format 2: ins(%src, %tmp {isBinary = ...}: type, type) outs(%dst : type)
    p << " ins(" << getSrc() << ", " << getTmp();
    // Print isBinary attribute if present
    SmallVector<StringRef, 1> elidedAttrs;
    if (!getIsBinaryAttr() || getIsBinaryAttr().getValue() == false) {
      elidedAttrs.push_back("isBinary");
    }
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    p << " : " << getSrc().getType() << ", " << getTmp().getType() << ")";
  } else {
    // Format 1: ins(%src : type) outs(%dst : type)
    p << " ins(" << getSrc() << " : " << getSrc().getType() << ")";
  }

  p << " outs(" << getDst() << " : " << getDst().getType() << ")";

  // Print remaining attributes for format 1 (excluding isBinary)
  if (!getTmp()) {
    SmallVector<StringRef, 1> elidedAttrs = {"isBinary"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  }
}

LogicalResult pto::TColSumOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
      return failure();
    bool hasTmp = (bool)getTmp();
    bool hasIsBinary = (bool)getIsBinaryAttr();
    if (hasTmp != hasIsBinary) {
      if (hasTmp)
        return emitOpError("tmp operand requires isBinary attribute");
      return emitOpError("isBinary attribute requires tmp operand");
    }
    if (getTmp()) {
      Type tmpTy = getTmp().getType();
      if (failed(verifyNDStyleVecTile(*this, tmpTy, "tmp")))
        return failure();
      if (getElemTy(srcTy) != getElemTy(dstTy) || getElemTy(srcTy) != getElemTy(tmpTy))
        return emitOpError("expects src/tmp/dst element types to match");
    }
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError("expects src/dst element types to match");
    if (failed(verifyColReductionValidRegion(*this, srcTy, dstTy,
                                             /*requireNonZeroSrc=*/false)))
      return failure();
    Type elem = getElemTy(srcTy);
    if (!(elem.isF16() || elem.isF32() || elem.isInteger(16) || elem.isInteger(32)))
      return emitOpError("expects A2/A3 tcolsum element type to be f16/f32/i16/i32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
      return failure();
    bool hasTmp = (bool)getTmp();
    bool hasIsBinary = (bool)getIsBinaryAttr();
    if (hasTmp != hasIsBinary) {
      if (hasTmp)
        return emitOpError("tmp operand requires isBinary attribute");
      return emitOpError("isBinary attribute requires tmp operand");
    }
    if (getTmp()) {
      Type tmpTy = getTmp().getType();
      if (failed(verifyNDStyleVecTile(*this, tmpTy, "tmp")))
        return failure();
      if (getElemTy(srcTy) != getElemTy(dstTy) || getElemTy(srcTy) != getElemTy(tmpTy))
        return emitOpError("expects src/tmp/dst element types to match");
    }
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError("expects src/dst element types to match");
    if (failed(verifyColReductionValidRegion(*this, srcTy, dstTy,
                                             /*requireNonZeroSrc=*/true)))
      return failure();
    Type elem = getElemTy(srcTy);
    if (!(elem.isF16() || elem.isF32() || elem.isBF16() || elem.isInteger(8) ||
          elem.isInteger(16) || elem.isInteger(32)))
      return emitOpError("expects A5 tcolsum element type to be i8/i16/i32/f16/bf16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TColProdOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/false,
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolprod element type to be f16/f32/i16/i32",
      "expects A5 tcolprod element type to be i16/ui16/i32/ui32/f16/bf16/f32");
}

llvm::LogicalResult mlir::pto::TCvtOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src", /*allowLowPrecision=*/true)) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst", /*allowLowPrecision=*/true)))
    return failure();
  if (failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/false)))
    return failure();
  if (failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/true)))
    return failure();
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))
      return emitOpError("expects A2/A3 tcvt low-precision element types to be unsupported");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!isA5SupportedTCvtPair(srcElem, dstElem))
      return emitOpError("expects A5 tcvt low-precision type pairs to match PTO-ISA support");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

llvm::LogicalResult mlir::pto::TRandomOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("trandom is only supported for A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (shouldBypassDecodedMemrefVerifier(getOperation()))
      return success();

    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (!isRowMajorTileBuf(dstTy))
      return emitOpError("expects dst to use row-major layout");

    Type elemTy = getElemTy(dstTy);
    if (!elemTy.isInteger(32))
      return emitOpError("expects dst element type to be i32 or ui32");

    auto checkWord = [&](Value v, StringRef name) -> LogicalResult {
      auto ty = dyn_cast<IntegerType>(v.getType());
      if (!ty || ty.getWidth() != 32)
        return emitOpError() << "expects " << name << " to be i32/ui32";
      return success();
    };
    if (failed(checkWord(getKey0(), "key0")) ||
        failed(checkWord(getKey1(), "key1")) ||
        failed(checkWord(getCounter0(), "counter0")) ||
        failed(checkWord(getCounter1(), "counter1")) ||
        failed(checkWord(getCounter2(), "counter2")) ||
        failed(checkWord(getCounter3(), "counter3")))
      return failure();

    int32_t rounds = getRounds();
    if (rounds != 7 && rounds != 10)
      return emitOpError("expects rounds to be 7 or 10");

    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TDivOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr))
      return failure();
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32()))
      return emitOpError("expects A2/A3 tdiv element type to be f16 or f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr))
      return failure();
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32() || elem0.isInteger(16) || elem0.isInteger(32)))
      return emitOpError("expects A5 tdiv element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDivSOp::verify() {
  auto isTileLike = [](Type ty) -> bool {
    return isa<mlir::pto::TileBufType, MemRefType, RankedTensorType,
               mlir::pto::PartitionTensorViewType>(ty);
  };
  auto isScalarLike = [](Type ty) -> bool {
    return mlir::isa<IntegerType, FloatType>(ty);
  };

  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type rhsTy = getScalar().getType();
    Type dstTy = getDst().getType();

    bool srcTile = isTileLike(srcTy);
    bool rhsTile = isTileLike(rhsTy);
    bool srcScalar = isScalarLike(srcTy);
    bool rhsScalar = isScalarLike(rhsTy);

    if (!(srcTile && rhsScalar) && !(srcScalar && rhsTile))
      return emitOpError("expects one tile-like operand and one scalar operand in ins(...)");

    Type tileTy = srcTile ? srcTy : rhsTy;
    Type scalarTy = srcTile ? rhsTy : srcTy;

    if (failed(verifyScalarTileOp(*this, tileTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true)))
      return failure();
    if (!mlir::isa<IntegerType, FloatType>(scalarTy))
      return emitOpError("scalar must be a scalar type (integer/float)");
    Type elem = getElemTy(tileTy);
    if (targetArch == PTOArch::A3 &&
        !(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32()))
      return emitOpError("expects A2/A3 tdivs element type to be i32/i16/f16/f32");
    if (targetArch == PTOArch::A5 &&
        !(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 tdivs element type to be i32/i16/i8/f16/f32");
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExpOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                    /*allowBf16=*/false, /*allowInt8=*/false)))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type srcElem = getElemTy(srcTy);
    if (!srcElem.isF16() && !srcElem.isF32())
      return emitOpError("expects element type to be f16 or f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExpandsOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT))
      return emitOpError("expects dst to be in the vec or mat address space");
    Type dstElem = getElemTy(dstTy);
    Type scalarTy = getScalar().getType();
    if (scalarTy != dstElem)
      return emitOpError("expects scalar type == dst element type");
    if (*dstSpace == pto::AddressSpace::VEC && !isRowMajorTileBuf(dstTy))
      return emitOpError("expects vec dst to use row-major layout on A2/A3");
    if (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32())
      return mlir::success();
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
      unsigned w = it.getWidth();
      if (w == 16 || w == 32)
        return mlir::success();
    }
    return emitOpError("expects A2/A3 texpands dst element type to be i16/i32/f16/bf16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT))
      return emitOpError("expects dst to be in the vec or mat address space");
    Type dstElem = getElemTy(dstTy);
    Type scalarTy = getScalar().getType();
    if (scalarTy != dstElem)
      return emitOpError("expects scalar type == dst element type");
    if (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32())
      return mlir::success();
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
      unsigned w = it.getWidth();
      if (w == 8 || w == 16 || w == 32)
        return mlir::success();
    }
    return emitOpError("expects A5 texpands dst element type to be i8/i16/i32/f16/bf16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExtractOp::verify() {
  auto hasMatExtractSourceLayoutA2A3 = [&](pto::TileBufType srcTy) -> bool {
    int32_t bl = srcTy.getBLayoutValueI32();
    int32_t sl = srcTy.getSLayoutValueI32();
    return bl == static_cast<int32_t>(pto::BLayout::RowMajor) ||
           (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::RowMajor));
  };
  auto hasMatExtractSourceLayoutA5 = [&](pto::TileBufType srcTy,
                                         pto::AddressSpace dstSpace) -> bool {
    int32_t bl = srcTy.getBLayoutValueI32();
    int32_t sl = srcTy.getSLayoutValueI32();
    if (dstSpace == pto::AddressSpace::LEFT) {
      return (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
              sl == static_cast<int32_t>(pto::SLayout::ColMajor)) ||
             (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
              sl == static_cast<int32_t>(pto::SLayout::RowMajor)) ||
             bl == static_cast<int32_t>(pto::BLayout::RowMajor);
    }
    return (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::ColMajor)) ||
           (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::RowMajor));
  };
  auto isA2A3ExtractElemType = [&](Type ty) -> bool {
    return ty.isInteger(8) || ty.isF16() || ty.isBF16() || ty.isF32();
  };
  auto isA5ExtractElemType = [&](Type ty) -> bool {
    if (auto it = dyn_cast<IntegerType>(ty))
      return it.getWidth() == 8;
    if (auto ft = dyn_cast<FloatType>(ty))
      return ft.getWidth() == 8 || ft.isF16() || ft.isBF16() || ft.isF32();
    return false;
  };
  auto isRowMajorNoneBoxND = [&](pto::TileBufType ty) -> bool {
    return ty.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor) &&
           ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::NoneBox);
  };
  auto verifyCommon = [&]() -> FailureOr<std::tuple<Type, Type, pto::TileBufType,
                                                    pto::TileBufType, Type, Type,
                                                    std::optional<pto::AddressSpace>,
                                                    std::optional<pto::AddressSpace>>> {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (!srcTb || !dstTb)
      return emitOpError("expects src and dst to be !pto.tile_buf");
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")) ||
        failed(verifyNonNegativeIndexRowCol(
            *getOperation(), getIndexRow(), getIndexCol(),
            /*includeIndexAndIntOpsInConstFold=*/false)) ||
        failed(verifyExtractStaticBoundsCommon(
            *getOperation(), getIndexRow(), getIndexCol(), srcTy, dstTy,
            /*includeIndexAndIntOpsInConstFold=*/false)))
      return failure();
    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem || srcElem != dstElem)
      return emitOpError("expects src and dst to have the same element type");
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    return std::make_tuple(srcTy, dstTy, srcTb, dstTb, srcElem, dstElem,
                           srcSpace, dstSpace);
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon();
    if (failed(common))
      return failure();
    auto [srcTy, dstTy, srcTb, dstTb, srcElem, dstElem, srcSpace, dstSpace] =
        *common;
    (void)srcTy;
    (void)dstTy;
    (void)srcElem;
    if (!isA2A3ExtractElemType(dstElem))
      return emitOpError("expects A2/A3 textract element type to be i8/f16/bf16/f32");
    if (srcSpace && dstSpace && *srcSpace == pto::AddressSpace::VEC &&
        *dstSpace == pto::AddressSpace::VEC)
      return mlir::success();
    if (!srcSpace || *srcSpace != pto::AddressSpace::MAT)
      return emitOpError("expects A2/A3 textract src to use loc=mat or vec");
    if (!dstSpace || (*dstSpace != pto::AddressSpace::LEFT &&
                      *dstSpace != pto::AddressSpace::RIGHT))
      return emitOpError("expects A2/A3 textract dst to use loc=left, loc=right, or loc=vec");
    if (!hasMatExtractSourceLayoutA2A3(srcTb))
      return emitOpError("expects A2/A3 textract src to use a supported mat blayout/slayout combination");
    if (*dstSpace == pto::AddressSpace::LEFT) {
      if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
          dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
        return emitOpError("expects A2/A3 left dst to use row_major blayout and row_major slayout");
    } else {
      if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
          dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor))
        return emitOpError("expects A2/A3 right dst to use row_major blayout and col_major slayout");
    }
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon();
    if (failed(common))
      return failure();
    auto [srcTy, dstTy, srcTb, dstTb, srcElem, dstElem, srcSpace, dstSpace] =
        *common;
    (void)srcTy;
    (void)dstTy;
    (void)srcElem;
    if (!isA5ExtractElemType(dstElem))
      return emitOpError("expects A5 textract element type to be an fp8/f16/bf16/f32 or int8 family type");
    if (!srcSpace || !dstSpace)
      return emitOpError("expects src and dst to have explicit loc");
    bool okPair =
        (*srcSpace == pto::AddressSpace::MAT &&
         (*dstSpace == pto::AddressSpace::LEFT ||
          *dstSpace == pto::AddressSpace::RIGHT ||
          *dstSpace == pto::AddressSpace::SCALING)) ||
        (*srcSpace == pto::AddressSpace::VEC &&
         (*dstSpace == pto::AddressSpace::MAT ||
          *dstSpace == pto::AddressSpace::VEC));
    if (!okPair)
      return emitOpError("expects A5 textract to use a supported src/dst loc pair");
    if (*srcSpace == pto::AddressSpace::MAT) {
      if (!hasMatExtractSourceLayoutA5(srcTb, *dstSpace))
        return emitOpError("expects A5 textract src to use a supported mat blayout/slayout combination");
      if (*dstSpace == pto::AddressSpace::LEFT) {
        if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
            dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
          return emitOpError("expects A5 left dst to use col_major blayout and row_major slayout");
      } else if (*dstSpace == pto::AddressSpace::RIGHT) {
        if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
          dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor))
          return emitOpError("expects A5 right dst to use row_major blayout and col_major slayout");
      }
    } else if (*srcSpace == pto::AddressSpace::VEC &&
               *dstSpace == pto::AddressSpace::VEC) {
      if (!isRowMajorNoneBoxND(srcTb) || !isRowMajorNoneBoxND(dstTb))
        return emitOpError(
            "expects A5 vec->vec textract src/dst to use ND layout "
            "(blayout=row_major, slayout=none_box)");
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem);
mlir::LogicalResult mlir::pto::TInsertOp::verify() {
  auto isColMajorRowMajorNZ = [&](pto::TileBufType ty) -> bool {
    return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
           ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
  };
  auto isRowMajorNoneBoxND = [&](pto::TileBufType ty) -> bool {
    return ty.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor) &&
           ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::NoneBox);
  };
  auto isA5SupportedVecElemType = [&](Type ty) -> bool {
    if (auto it = dyn_cast<IntegerType>(ty))
      return it.getWidth() == 8 || it.getWidth() == 32;
    if (auto ft = dyn_cast<FloatType>(ty))
      return ft.getWidth() == 8 || ft.isF16() || ft.isBF16() || ft.isF32();
    return false;
  };
  auto isA2A3VecInsertElemType = [&](Type ty) -> bool {
    return ty.isInteger(8) || ty.isF16() || ty.isBF16() || ty.isF32();
  };
  auto getSpace = [](Type ty) -> std::optional<pto::AddressSpace> {
    return getPTOMemorySpaceEnum(ty);
  };
  auto verifyCommon = [&]() -> FailureOr<std::tuple<Type, Type, pto::TileBufType,
                                                    pto::TileBufType, Type, Type,
                                                    std::optional<pto::AddressSpace>,
                                                    std::optional<pto::AddressSpace>>> {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (!srcTb || !dstTb)
      return emitOpError("expects src and dst to be !pto.tile_buf");
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")) ||
        failed(verifyNonNegativeIndexRowCol(
            *getOperation(), getIndexRow(), getIndexCol(),
            /*includeIndexAndIntOpsInConstFold=*/true)) ||
        failed(verifyInsertStaticBoundsCommon(
            *getOperation(), getIndexRow(), getIndexCol(), srcTy, dstTy,
            /*includeIndexAndIntOpsInConstFold=*/true)))
      return failure();
    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    return std::make_tuple(srcTy, dstTy, srcTb, dstTb, srcElem, dstElem,
                           srcSpace, dstSpace);
  };
  // Shared validation for optional operands and attributes.
  auto verifyOptionalArgs = [&](const std::optional<pto::AddressSpace> &srcSpace,
                                const std::optional<pto::AddressSpace> &dstSpace,
                                bool isA5) -> LogicalResult {
    const bool hasFp = static_cast<bool>(getFp());
    const bool hasPreQuantScalar = static_cast<bool>(getPreQuantScalar());
    const bool hasAccToVecMode = static_cast<bool>(getAccToVecModeAttr());
    const bool hasInsertMode = static_cast<bool>(getTinsertModeAttr());
    const bool reluNonDefault = getReluPreMode() != pto::ReluPreMode::NoRelu;

    if (hasFp && hasPreQuantScalar)
      return emitOpError("fp and preQuantScalar are mutually exclusive");

    // fp tile is only valid with Acc source.
    if (hasFp) {
      if (!srcSpace || *srcSpace != pto::AddressSpace::ACC)
        return emitOpError("fp is only valid with src loc=acc");
      auto fpTy = getFp().getType();
      auto fpTb = dyn_cast<pto::TileBufType>(fpTy);
      if (!fpTb) return emitOpError("expects fp to be !pto.tile_buf");
      auto fpSpace = getSpace(fpTy);
      if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING)
        return emitOpError("expects fp to be loc=scaling");
    }

    // preQuantScalar is only valid with Acc source.
    if (hasPreQuantScalar) {
      if (!srcSpace || *srcSpace != pto::AddressSpace::ACC)
        return emitOpError("preQuantScalar is only valid with src loc=acc");
    }

    // reluPreMode is only valid with Acc source.
    if (reluNonDefault) {
      if (!srcSpace || *srcSpace != pto::AddressSpace::ACC)
        return emitOpError("reluPreMode is only valid with src loc=acc");
    }

    // accToVecMode is only valid with Acc->Vec (A5 only).
    if (hasAccToVecMode) {
      if (!isA5)
        return emitOpError("accToVecMode is only supported on A5");
      if (!srcSpace || !dstSpace ||
          *srcSpace != pto::AddressSpace::ACC ||
          *dstSpace != pto::AddressSpace::VEC)
        return emitOpError("accToVecMode is only valid with src=acc, dst=vec");
    }

    // tinsertMode (SPLIT2/SPLIT4) is only valid with Vec(NZ)->Mat on A5.
    if (hasInsertMode) {
      if (!isA5)
        return emitOpError("tinsertMode is only supported on A5");
      if (!srcSpace || !dstSpace ||
          *srcSpace != pto::AddressSpace::VEC ||
          *dstSpace != pto::AddressSpace::MAT)
        return emitOpError(
            "tinsertMode (SPLIT2/SPLIT4) is only valid with src=vec, dst=mat");
      auto srcTb = dyn_cast<pto::TileBufType>(getSrc().getType());
      if (!srcTb || !isColMajorRowMajorNZ(srcTb))
        return emitOpError(
            "tinsertMode (SPLIT2/SPLIT4) requires src NZ layout "
            "(blayout=col_major, slayout=row_major)");
    }

    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon();
    if (failed(common))
      return failure();
    auto [srcTy, dstTy, srcTb, dstTb, srcElem, dstElem, srcSpace, dstSpace] =
        *common;
    if (failed(verifyOptionalArgs(srcSpace, dstSpace, /*isA5=*/false)))
      return failure();
    if (srcSpace && dstSpace && *srcSpace == pto::AddressSpace::VEC &&
        *dstSpace == pto::AddressSpace::VEC) {
      if (srcElem != dstElem || !isA2A3VecInsertElemType(srcElem))
        return emitOpError(
            "expects A2/A3 vec->vec tinsert src/dst to have same supported dtype "
            "(i8/f16/bf16/f32)");
      return success();
    }
    if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::ACC ||
        *dstSpace != pto::AddressSpace::MAT)
      return emitOpError("expects A2/A3 tinsert to use acc->mat or vec->vec");

    if (!isColMajorRowMajorNZ(srcTb))
      return emitOpError("expects A2/A3 tinsert src to use blayout=col_major and slayout=row_major");
    if (!isColMajorRowMajorNZ(dstTb))
      return emitOpError("expects A2/A3 tinsert dst to use blayout=col_major and slayout=row_major");
    if (dstTb.getSFractalSizeI32() != 512)
      return emitOpError("expects A2/A3 tinsert dst fractal size to be 512");

    if (!(srcElem.isF32() && (dstElem.isF16() || dstElem.isBF16())))
      return emitOpError("expects A2/A3 tinsert element types to be src=f32, dst=f16/bf16");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon();
    if (failed(common))
      return failure();
    auto [srcTy, dstTy, srcTb, dstTb, srcElem, dstElem, srcSpace, dstSpace] =
        *common;
    if (!srcSpace || !dstSpace)
      return emitOpError("expects A5 tinsert src/dst to have explicit loc");
    if (failed(verifyOptionalArgs(srcSpace, dstSpace, /*isA5=*/true)))
      return failure();

    // A5 regular acc->mat path.
    if (*srcSpace == pto::AddressSpace::ACC && *dstSpace == pto::AddressSpace::MAT) {
      if (!isColMajorRowMajorNZ(srcTb))
        return emitOpError("expects A5 acc->mat tinsert src to use blayout=col_major and slayout=row_major");
      if (!isColMajorRowMajorNZ(dstTb))
        return emitOpError("expects A5 acc->mat tinsert dst to use blayout=col_major and slayout=row_major");
      const bool hasQuant = static_cast<bool>(getFp()) ||
                            static_cast<bool>(getPreQuantScalar());
      bool okTypes;
      if (hasQuant) {
        // With fp/scalar quantization, allow wider dst types (i8/fp8/f16/bf16/f32).
        okTypes = isA5VectorPreQuantTypePair(srcElem, dstElem);
      } else {
        okTypes = (srcElem.isF32() &&
                   (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32())) ||
                  (srcElem.isInteger(32) && dstElem.isInteger(32));
      }
      if (!okTypes)
        return emitOpError(
            "expects A5 acc->mat tinsert element types to be "
            "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)"
            + (hasQuant ? std::string("; with fp/scalar: (src=f32,dst=i8/fp8/f16/bf16/f32) or (src=i32,dst=i8/f16/bf16)") : std::string()));
      return success();
    }

    // A5 acc->vec path.
    if (*srcSpace == pto::AddressSpace::ACC && *dstSpace == pto::AddressSpace::VEC) {
      if (!isColMajorRowMajorNZ(srcTb))
        return emitOpError("expects A5 acc->vec tinsert src to use blayout=col_major and slayout=row_major");
      bool dstIsND = isRowMajorNoneBoxND(dstTb);
      bool dstIsNZ = isColMajorRowMajorNZ(dstTb);
      if (!dstIsND && !dstIsNZ)
        return emitOpError(
            "expects A5 acc->vec tinsert dst to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
      const bool hasQuant = static_cast<bool>(getFp()) ||
                            static_cast<bool>(getPreQuantScalar());
      bool okTypes;
      if (hasQuant) {
        okTypes = isA5VectorPreQuantTypePair(srcElem, dstElem);
      } else {
        okTypes = (srcElem.isF32() &&
                   (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32())) ||
                  (srcElem.isInteger(32) && dstElem.isInteger(32));
      }
      if (!okTypes)
        return emitOpError(
            "expects A5 acc->vec tinsert element types to be "
            "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)"
            + (hasQuant ? std::string("; with fp/scalar: (src=f32,dst=i8/fp8/f16/bf16/f32) or (src=i32,dst=i8/f16/bf16)") : std::string()));
      return success();
    }

    // A5 vec->mat path (ND/NZ modes in pto-isa).
    if (*srcSpace == pto::AddressSpace::VEC && *dstSpace == pto::AddressSpace::MAT) {
      if (!isColMajorRowMajorNZ(dstTb))
        return emitOpError("expects A5 vec->mat tinsert dst to use blayout=col_major and slayout=row_major");
      bool srcIsND = isRowMajorNoneBoxND(srcTb);
      bool srcIsNZ = isColMajorRowMajorNZ(srcTb);
      if (!srcIsND && !srcIsNZ)
        return emitOpError(
            "expects A5 vec->mat tinsert src to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
      if (srcElem != dstElem || !isA5SupportedVecElemType(srcElem))
        return emitOpError(
            "expects A5 vec->mat tinsert src/dst to have same supported dtype "
            "(fp8/f16/bf16/f32/i8/i32)");
      return success();
    }

    // A5 vec->vec path: supports ND->ND and NZ->NZ.
    if (*srcSpace == pto::AddressSpace::VEC && *dstSpace == pto::AddressSpace::VEC) {
      bool srcIsND = isRowMajorNoneBoxND(srcTb);
      bool dstIsND = isRowMajorNoneBoxND(dstTb);
      bool srcIsNZ = isColMajorRowMajorNZ(srcTb);
      bool dstIsNZ = isColMajorRowMajorNZ(dstTb);
      if (srcIsND && dstIsND) {
        // ND->ND path
      } else if (srcIsNZ && dstIsNZ) {
        // NZ->NZ path
      } else {
        return emitOpError(
            "expects A5 vec->vec tinsert src/dst layouts to match: "
            "both ND(row_major/none_box) or both NZ(col_major/row_major)");
      }
      if (srcElem != dstElem || !isA5SupportedVecElemType(srcElem))
        return emitOpError(
            "expects A5 vec->vec tinsert src/dst to have same supported dtype "
            "(fp8/f16/bf16/f32/i8/i32)");
      return success();
    }

    return emitOpError(
        "expects A5 tinsert to use a supported src/dst loc pair: "
        "acc->mat, acc->vec, vec->mat, or vec->vec");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static bool isColMajorRowMajorNZTileBuf(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
}

static bool isA2A3VectorPreQuantTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32())
    return dstElem.isInteger(8);
  if (srcElem.isInteger(32))
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isInteger(16);
  return false;
}

static bool isA5Fp8LikeType(Type ty) {
  if (auto ft = dyn_cast<FloatType>(ty))
    return ft.getWidth() == 8;
  return false;
}

static bool isA5MxInputType(Type ty) {
  if (auto ft = dyn_cast<FloatType>(ty))
    return ft.isFloat8E4M3FN() || ft.isFloat8E5M2();
  return false;
}

static LogicalResult verifyA5MxTypeTriple(Operation *op, Type lhsTy, Type rhsTy,
                                          Type dstTy, StringRef lhsName,
                                          StringRef rhsName, StringRef dstName) {
  Type lhsElem = getElemTy(lhsTy);
  Type rhsElem = getElemTy(rhsTy);
  Type dstElem = getElemTy(dstTy);

  if (!isA5MxInputType(lhsElem))
    return op->emitOpError() << lhsName << ": dtype " << lhsElem
                             << " is not supported by this op yet";
  if (!isA5MxInputType(rhsElem))
    return op->emitOpError() << rhsName << ": dtype " << rhsElem
                             << " is not supported by this op yet";

  if (!dstElem.isF32())
    return op->emitOpError()
           << "expects A5 mx result " << dstName << " to use f32 element type";

  return success();
}

static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32())
    return dstElem.isInteger(8) || isA5Fp8LikeType(dstElem) || dstElem.isF16() ||
           dstElem.isBF16() || dstElem.isF32();
  if (srcElem.isInteger(32))
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16();
  return false;
}

mlir::LogicalResult mlir::pto::TExtractFPOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<std::tuple<Type, Type, Type, pto::TileBufType,
                                                    pto::TileBufType, pto::TileBufType,
                                                    pto::AddressSpace, pto::AddressSpace,
                                                    pto::AddressSpace>> {
    Type srcTy = getSrc().getType();
    Type fpTy = getFp().getType();
    Type dstTy = getDst().getType();
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto fpTb = dyn_cast<pto::TileBufType>(fpTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (!srcTb || !fpTb || !dstTb)
      return emitOpError("expects src, fp, and dst to be !pto.tile_buf");
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")) ||
        failed(verifyNonNegativeIndexRowCol(
            *getOperation(), getIndexRow(), getIndexCol(),
            /*includeIndexAndIntOpsInConstFold=*/true)) ||
        failed(verifyExtractStaticBoundsCommon(
            *getOperation(), getIndexRow(), getIndexCol(), srcTy, dstTy,
            /*includeIndexAndIntOpsInConstFold=*/true)))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || !fpSpace || !dstSpace)
      return emitOpError("expects src, fp, and dst to have explicit loc");
    if (*srcSpace != pto::AddressSpace::ACC)
      return emitOpError("expects src to use loc=acc");
    if (*fpSpace != pto::AddressSpace::SCALING)
      return emitOpError("expects fp to use loc=scaling");
    if (*dstSpace != pto::AddressSpace::MAT)
      return emitOpError("expects dst to use loc=mat");
    if (!isColMajorRowMajorNZTileBuf(srcTb))
      return emitOpError("expects src to use blayout=col_major and slayout=row_major");
    if (!isColMajorRowMajorNZTileBuf(dstTb))
      return emitOpError("expects dst to use blayout=col_major and slayout=row_major");
    return std::make_tuple(srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, *srcSpace,
                           *fpSpace, *dstSpace);
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon();
    if (failed(common))
      return failure();
    auto [srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, srcSpace, fpSpace, dstSpace] =
        *common;
    (void)fpTy;
    (void)srcSpace;
    (void)fpSpace;
    (void)dstSpace;
    if (dstTb.getSFractalSizeI32() != 512)
      return emitOpError("expects dst fractal size to be 512");
    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!isA2A3VectorPreQuantTypePair(srcElem, dstElem))
      return emitOpError(
          "expects A2/A3 textract_fp element types to be (src=f32,dst=i8) "
          "or (src=i32,dst=i8/f16/i16)");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon();
    if (failed(common))
      return failure();
    auto [srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, srcSpace, fpSpace, dstSpace] =
        *common;
    (void)fpTy;
    (void)srcTb;
    (void)fpTb;
    (void)dstTb;
    (void)srcSpace;
    (void)fpSpace;
    (void)dstSpace;
    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!isA5VectorPreQuantTypePair(srcElem, dstElem))
      return emitOpError(
          "expects A5 textract_fp element types to be (src=f32,dst=i8/fp8/f16/bf16/f32) "
          "or (src=i32,dst=i8/f16/bf16)");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TInsertFPOp::verify() {
  auto verifyCommon = [&](bool isA5) -> FailureOr<std::tuple<Type, Type, Type, pto::TileBufType,
                                                    pto::TileBufType, pto::TileBufType,
                                                    pto::AddressSpace, pto::AddressSpace,
                                                    pto::AddressSpace>> {
    Type srcTy = getSrc().getType();
    Type fpTy = getFp().getType();
    Type dstTy = getDst().getType();
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto fpTb = dyn_cast<pto::TileBufType>(fpTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (!srcTb || !fpTb || !dstTb)
      return emitOpError("expects src, fp, and dst to be !pto.tile_buf");
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")) ||
        failed(verifyNonNegativeIndexRowCol(
            *getOperation(), getIndexRow(), getIndexCol(),
            /*includeIndexAndIntOpsInConstFold=*/true)) ||
        failed(verifyInsertStaticBoundsCommon(
            *getOperation(), getIndexRow(), getIndexCol(), srcTy, dstTy,
            /*includeIndexAndIntOpsInConstFold=*/true)))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || !fpSpace || !dstSpace)
      return emitOpError("expects src, fp, and dst to have explicit loc");
    if (*srcSpace != pto::AddressSpace::ACC)
      return emitOpError("expects src to use loc=acc");
    if (*fpSpace != pto::AddressSpace::SCALING)
      return emitOpError("expects fp to use loc=scaling");
    // A2/A3: only acc->mat; A5: acc->mat or acc->vec.
    if (*dstSpace != pto::AddressSpace::MAT &&
        !(isA5 && *dstSpace == pto::AddressSpace::VEC))
      return emitOpError("expects dst to use loc=mat" +
                         (isA5 ? StringRef(" or loc=vec (A5)") : StringRef("")));
    if (!isColMajorRowMajorNZTileBuf(srcTb))
      return emitOpError("expects src to use blayout=col_major and slayout=row_major");
    if (*dstSpace == pto::AddressSpace::MAT && !isColMajorRowMajorNZTileBuf(dstTb))
      return emitOpError("expects dst (mat) to use blayout=col_major and slayout=row_major");
    // accToVecMode is only valid when dst=vec.
    if (static_cast<bool>(getAccToVecModeAttr()) &&
        *dstSpace != pto::AddressSpace::VEC)
      return emitOpError("accToVecMode is only valid with dst=vec");
    return std::make_tuple(srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, *srcSpace,
                           *fpSpace, *dstSpace);
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*isA5=*/false);
    if (failed(common))
      return failure();
    auto [srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, srcSpace, fpSpace, dstSpace] =
        *common;
    (void)fpTy;
    (void)srcTb;
    (void)fpTb;
    (void)srcSpace;
    (void)fpSpace;
    (void)dstSpace;
    if (dstTb.getSFractalSizeI32() != 512)
      return emitOpError("expects dst fractal size to be 512");
    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!isA2A3VectorPreQuantTypePair(srcElem, dstElem))
      return emitOpError(
          "expects A2/A3 tinsert_fp element types to be (src=f32,dst=i8) "
          "or (src=i32,dst=i8/f16/i16)");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*isA5=*/true);
    if (failed(common))
      return failure();
    auto [srcTy, fpTy, dstTy, srcTb, fpTb, dstTb, srcSpace, fpSpace, dstSpace] =
        *common;
    (void)fpTy;
    (void)srcTb;
    (void)fpTb;
    (void)dstTb;
    (void)srcSpace;
    (void)fpSpace;
    (void)dstSpace;
    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!isA5VectorPreQuantTypePair(srcElem, dstElem))
      return emitOpError(
          "expects A5 tinsert_fp element types to be (src=f32,dst=i8/fp8/f16/bf16/f32) "
          "or (src=i32,dst=i8/f16/bf16)");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static mlir::LogicalResult verifyTFillPadLike(Operation *op, Type srcTy, Type dstTy,
                                              bool allowDstExpand,
                                              llvm::StringRef opName) {
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy))
    return op->emitError("expects src/dst to be PTO shaped-like types");

  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2)
    return op->emitError("expects rank-2 shaped types for src/dst");

  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);

  auto getElemBytes = [](mlir::Type t) -> int64_t {
    unsigned elemBytes = getPTOStorageElemByteSize(t);
    return elemBytes == 0 ? -1 : static_cast<int64_t>(elemBytes);
  };

  int64_t srcB = getElemBytes(srcElem);
  int64_t dstB = getElemBytes(dstElem);
  if (srcB < 0 || dstB < 0)
    return op->emitError("unsupported element type (expects int/float element types)");
  if (srcB != dstB)
    return op->emitError("expects sizeof(src element) == sizeof(dst element)");
  if (!(srcB == 1 || srcB == 2 || srcB == 4))
    return op->emitError("expects element size to be 1, 2, or 4 bytes");

  // pto.tfillpad lowers to TFILLPAD(dst, src). For loc=mat, pto-isa only
  // exposes the homogeneous overload, so src/dst must use the same Tile<...>
  // specialization (including valid_shape and pad).
  // Note: tfillpad_expand is intentionally not covered here because its
  // cross-layer ABI contract for loc=mat heterogeneous shape expansion is not
  // finalized yet.
  if (opName == "tfillpad") {
    auto srcTb = mlir::dyn_cast<mlir::pto::TileBufType>(srcTy);
    auto dstTb = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy);
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (srcTb && dstTb && srcSpace && dstSpace &&
        *srcSpace == mlir::pto::AddressSpace::MAT &&
        *dstSpace == mlir::pto::AddressSpace::MAT && srcTb != dstTb) {
      auto dimToStr = [](int64_t dim) -> std::string {
        return dim == ShapedType::kDynamic ? "?" : std::to_string(dim);
      };
      SmallVector<std::string, 4> mismatchFields;
      auto srcValid = getValidShapeVec(srcTy);
      auto dstValid = getValidShapeVec(dstTy);
      if (srcValid.size() == 2 && dstValid.size() == 2) {
        if (srcValid[0] != dstValid[0])
          mismatchFields.push_back("v_row (" + dimToStr(srcValid[0]) + " vs " +
                                   dimToStr(dstValid[0]) + ")");
        if (srcValid[1] != dstValid[1])
          mismatchFields.push_back("v_col (" + dimToStr(srcValid[1]) + " vs " +
                                   dimToStr(dstValid[1]) + ")");
      }
      if (srcTb.getPadValueI32() != dstTb.getPadValueI32())
        mismatchFields.push_back("pad (" + std::to_string(srcTb.getPadValueI32()) +
                                 " vs " + std::to_string(dstTb.getPadValueI32()) +
                                 ")");

      auto diag = op->emitError()
                  << "expects src/dst tile types to be lowerable to TFILLPAD "
                     "for loc=mat";
      if (!mismatchFields.empty())
        diag << "; mismatching fields: " << llvm::join(mismatchFields, ", ");
      diag << "\n  src: " << srcTy;
      diag << "\n  dst: " << dstTy;
      diag << "\n  note: heterogeneous TFILLPAD overload is only available for loc=vec";
      return failure();
    }
  }

  if (auto dstTileTy = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy)) {
    auto padAttr = mlir::dyn_cast<mlir::pto::PadValueAttr>(dstTileTy.getPadValueAttr());
    if (!padAttr || padAttr.getValue() == mlir::pto::PadValue::Null)
      return op->emitError() << "expects dst PadVal != Null for " << opName;
  }

  if (!allowDstExpand) {
    if (srcShape != dstShape)
      return op->emitError()
             << "expects src and dst to have the same static shape for " << opName;
    return mlir::success();
  }

  if (srcShape[0] > dstShape[0] || srcShape[1] > dstShape[1]) {
    return op->emitError()
           << "expects dst static shape to be >= src static shape for " << opName;
  }

  return mlir::success();
}

mlir::LogicalResult mlir::pto::TFillPadOp::verify() {
  return verifyTFillPadLike(getOperation(), getSrc().getType(), getDst().getType(),
                            /*allowDstExpand=*/false, "tfillpad");
}

mlir::LogicalResult mlir::pto::TFillPadExpandOp::verify() {
  return verifyTFillPadLike(getOperation(), getSrc().getType(), getDst().getType(),
                            /*allowDstExpand=*/true, "tfillpad_expand");
}

mlir::LogicalResult mlir::pto::TFillPadInplaceOp::verify() {
  return verifyTFillPadLike(getOperation(), getSrc().getType(), getDst().getType(),
                            /*allowDstExpand=*/false, "tfillpad_inplace");
}


llvm::LogicalResult mlir::pto::TGatherOp::verify() {
  auto isSupportedGatherElemTypeA5Index = [&](Type ty) -> bool {
    if (ty.isF16() || ty.isF32())
      return true;
    if (auto it = dyn_cast<IntegerType>(ty)) {
      unsigned width = it.getWidth();
      return width == 8 || width == 16 || width == 32;
    }
    return false;
  };

  auto verifyMaskForm = [&](bool allowA5MaskTypes) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem)
      return emitOpError("failed to get element type for src/dst");
    if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy))
      return emitOpError("expects src and dst to use row-major layout");
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
        *dstSpace != pto::AddressSpace::VEC)
      return emitOpError("expects src and dst to be in the vec address space");
    unsigned srcElemBytes = getPTOStorageElemByteSize(srcElem);
    unsigned dstElemBytes = getPTOStorageElemByteSize(dstElem);
    if (srcElemBytes == 0 || dstElemBytes == 0)
      return emitOpError("failed to get element size for src/dst");
    if (srcElemBytes != dstElemBytes)
      return emitOpError("expects src and dst element sizes to match");

    auto dstValid = getValidShapeVec(dstTy);
    auto dstShape = getShapeVec(dstTy);
    if (dstValid.size() == 2 && dstShape.size() == 2 &&
        dstValid[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        dstValid[1] != dstShape[1]) {
      return emitOpError("expects dst valid_shape[1] to equal dst cols");
    }

    if (allowA5MaskTypes) {
      if (!(srcElemBytes == 1 || srcElemBytes == 2 || srcElemBytes == 4))
        return emitOpError("expects A5 mask-pattern gather element size to be 1, 2, or 4 bytes");
      if (!isSupportedGatherElemTypeA5(srcElem) || !isSupportedGatherElemTypeA5(dstElem))
        return emitOpError(
            "expects A5 mask-pattern gather src/dst element type to be i8/i16/i32/f16/bf16/f32/fp8-like");
    } else {
      if (!(srcElemBytes == 2 || srcElemBytes == 4))
        return emitOpError("expects A2/A3 mask-pattern gather element size to be 2 or 4 bytes");
    }
    return success();
  };

  auto verifyIndexForm = [&](bool allow16BitIndices, bool allowA5ElemTypes) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    Type idxTy = getIndices().getType();
    Type tmpTy = getTmp().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")) ||
        failed(verifyTileBufCommon(*this, idxTy, "indices")) ||
        failed(verifyTileBufCommon(*this, tmpTy, "tmp")))
      return failure();

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem)
      return emitOpError("failed to get element type for src/dst");
    if (srcElem != dstElem)
      return emitOpError("expects src and dst to have the same element type");
    if (allowA5ElemTypes) {
      if (!isSupportedGatherElemTypeA5Index(srcElem) ||
          !isSupportedGatherElemTypeA5Index(dstElem))
        return emitOpError(
            "expects A5 gather src/dst element type to be i8/i16/i32/f16/f32");
    } else if (!isSupportedGatherElemTypeA2A3(srcElem) ||
               !isSupportedGatherElemTypeA2A3(dstElem)) {
      return emitOpError("expects gather src/dst element type to be i16/i32/f16/f32");
    }

    auto idxElem = dyn_cast<IntegerType>(getElemTy(idxTy));
    if (!idxElem)
      return emitOpError("indices element type must be integer");
    unsigned width = idxElem.getWidth();
    if (!(width == 32 || (allow16BitIndices && width == 16))) {
      return emitOpError() << "expects indices element type to be i32"
                           << (allow16BitIndices ? " or i16" : "");
    }

    auto dstValid = getValidShapeVec(dstTy);
    auto dstShape = getShapeVec(dstTy);
    if (dstValid.size() == 2 && dstShape.size() == 2 &&
        dstValid[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        dstValid[1] != dstShape[1]) {
      return emitOpError("expects dst valid_shape[1] to equal dst cols");
    }

    auto idxValid = getValidShapeVec(idxTy);
    auto idxShape = getShapeVec(idxTy);
    if (idxValid.size() == 2 && idxShape.size() == 2 &&
        idxValid[1] != ShapedType::kDynamic && idxShape[1] != ShapedType::kDynamic &&
        idxValid[1] != idxShape[1]) {
      return emitOpError("expects indices valid_shape[1] to equal indices cols");
    }

    if (!allowA5ElemTypes) {
      Type tmpElem = getElemTy(tmpTy);
      if (tmpElem != idxElem)
        return emitOpError("expects tmp and indices to have the same element type");
      if (failed(verifyTileBufSameValidShape(*this, idxTy, tmpTy, "indices", "tmp")))
        return failure();
    }
    return success();
  };

  auto verifyCompareForm = [&](bool allowA5SrcTypes) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    Type cdstTy = getCdst().getType();
    Type tmpTy = getTmp().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")) ||
        failed(verifyTileBufCommon(*this, cdstTy, "cdst")) ||
        failed(verifyTileBufCommon(*this, tmpTy, "tmp")))
      return failure();

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    Type cdstElem = getElemTy(cdstTy);
    if (!srcElem || !dstElem || !cdstElem)
      return emitOpError("failed to get element type for src/dst/cdst");
    auto dstInt = dyn_cast<IntegerType>(dstElem);
    if (!dstInt || dstInt.getWidth() != 32)
      return emitOpError("expects dst element type to be i32");
    if (cdstElem != dstElem)
      return emitOpError("expects cdst to have the same element type as dst");
    if (getKValue().getType() != srcElem)
      return emitOpError("expects kValue to have the same type as src element type");

    auto cmpAttr = getCmpModeAttr();
    auto cmpMode = cmpAttr ? cmpAttr.getValue() : pto::CmpMode::EQ;
    if (cmpMode != pto::CmpMode::EQ && cmpMode != pto::CmpMode::GT)
      return emitOpError("expects compare-form tgather cmpMode to be eq or gt");

    if (allowA5SrcTypes) {
      if (!(srcElem.isF16() || srcElem.isF32() || srcElem.isInteger(16) ||
            srcElem.isInteger(32))) {
        return emitOpError(
            "expects A5 compare-form tgather src element type to be i16/i32/f16/f32");
      }
    } else {
      if (!(srcElem.isF16() || srcElem.isF32() ||
            (srcElem.isInteger(32) && cmpMode == pto::CmpMode::EQ))) {
        return emitOpError(
            "expects A2/A3 compare-form tgather src element type to be f16/f32, or i32 when cmpMode=eq");
      }
    }

    if (failed(verifyVecTileCommonA2A3(*this, srcTy, "src")) ||
        failed(verifyVecTileCommonA2A3(*this, dstTy, "dst")) ||
        failed(verifyVecTileCommonA2A3(*this, cdstTy, "cdst")) ||
        failed(verifyVecTileCommonA2A3(*this, tmpTy, "tmp")))
      return failure();
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (getMaskPatternAttr()) {
      if (getCdst() || getIndices() || getTmp() || getKValue())
        return emitOpError("mask-pattern tgather only allows src and dst operands");
      return verifyMaskForm(/*allowA5MaskTypes=*/false);
    }
    if (getCdst() || getKValue()) {
      if (!getCdst() || !getKValue() || !getTmp())
        return emitOpError("compare-form tgather expects dst, cdst, kValue, and tmp");
      if (getIndices())
        return emitOpError("compare-form tgather does not take indices");
      return verifyCompareForm(/*allowA5SrcTypes=*/false);
    }
    if (!getIndices() || !getTmp())
      return emitOpError("index-form tgather expects both indices and tmp");
    return verifyIndexForm(/*allow16BitIndices=*/false, /*allowA5ElemTypes=*/false);
  };

  auto verifyA5 = [&]() -> LogicalResult {
    if (getMaskPatternAttr()) {
      if (getCdst() || getIndices() || getTmp() || getKValue())
        return emitOpError("mask-pattern tgather only allows src and dst operands");
      return verifyMaskForm(/*allowA5MaskTypes=*/true);
    }
    if (getCdst() || getKValue()) {
      if (!getCdst() || !getKValue() || !getTmp())
        return emitOpError("compare-form tgather expects dst, cdst, kValue, and tmp");
      if (getIndices())
        return emitOpError("compare-form tgather does not take indices");
      return verifyCompareForm(/*allowA5SrcTypes=*/true);
    }
    if (!getIndices() || !getTmp())
      return emitOpError("index-form tgather expects both indices and tmp");
    return verifyIndexForm(/*allow16BitIndices=*/true, /*allowA5ElemTypes=*/true);
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
mlir::LogicalResult mlir::pto::TGatherBOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<std::pair<Type, Type>> {
    Type srcTy = getSrc().getType();
    Type offTy = getOffsets().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, offTy, "offsets")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto srcElemTy = getElemTy(srcTy);
    auto dstElemTy = getElemTy(dstTy);
    if (!srcElemTy || !dstElemTy)
      return emitOpError() << "failed to get element type for src/dst";
    return std::make_pair(srcElemTy, dstElemTy);
  };

  auto getElemBytes = [](Type ty) -> std::optional<unsigned> {
    unsigned elemBytes = getPTOStorageElemByteSize(ty);
    if (elemBytes == 0)
      return std::nullopt;
    return elemBytes;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<std::pair<Type, Type>> elems = verifyCommon();
    if (failed(elems))
      return failure();
    Type dstTy = getDst().getType();
    Type dstElemTy = elems->second;
    if (!isRowMajorTileBuf(dstTy))
      return emitOpError() << "expects dst to use row-major layout";
    auto dstBytes = getElemBytes(dstElemTy);
    if (!dstBytes || (*dstBytes != 1 && *dstBytes != 2 && *dstBytes != 4))
      return emitOpError() << "expects dst element size to be 1, 2, or 4 bytes";
    return mlir::success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<std::pair<Type, Type>> elems = verifyCommon();
    if (failed(elems))
      return failure();
    Type dstElemTy = elems->second;
    auto dstBytes = getElemBytes(dstElemTy);
    if (!dstBytes || (*dstBytes != 1 && *dstBytes != 2 && *dstBytes != 4))
      return emitOpError() << "expects dst element size to be 1, 2, or 4 bytes";
    return mlir::success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TLogOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();
  auto elemTy = getElemTy(srcTy);
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";
  return success();
}

mlir::LogicalResult mlir::pto::TLReluOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    auto valid = getValidShapeVec(srcTy);
    if (valid.size() != 2)
      return emitOpError("expects src to have rank-2 valid_shape");
    if (valid[0] != ShapedType::kDynamic && valid[0] < 0)
      return emitOpError("expects src valid_shape[0] to be non-negative");
    if (valid[1] != ShapedType::kDynamic && valid[1] < 0)
      return emitOpError("expects src valid_shape[1] to be non-negative");
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << "expects A2/A3 tlrelu element type to be f16 or f32";
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << "expects A5 tlrelu element type to be f16 or f32";
    if (!getSlope().getType().isF32())
      return emitOpError() << "expects slope to have type f32";
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMaxOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/false,
      "expects A2/A3 tmax element type to be i32/i16/f16/f32",
      "expects A5 tmax element type to be i32/i16/i8/f16/f32");
}

mlir::LogicalResult mlir::pto::TMaxSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmaxs element type to be i32/i16/f16/f32",
      "expects A5 tmaxs element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

mlir::LogicalResult mlir::pto::TMinOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmin element type to be i32/i16/f16/f32",
      "expects A5 tmin element type to be i32/i16/i8/f16/bf16/f32");
}

mlir::LogicalResult mlir::pto::TMinSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmins element type to be i32/i16/f16/f32",
      "expects A5 tmins element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

mlir::LogicalResult mlir::pto::TMovOp::verify() {
  auto verifyImpl = [&](bool isA5) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    Value fp = getFp();
    Value preQuantScalar = getPreQuantScalar();
    auto accToVecModeAttr = getAccToVecModeAttr();
    auto reluMode = getReluPreMode();
    const bool hasFp = static_cast<bool>(fp);
    const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);

    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (hasFp && failed(verifyTileBufCommon(*this, fp.getType(), "fp")))
      return failure();
    if (hasFp && hasPreQuantScalar)
      return emitOpError() << "expects fp and preQuantScalar forms to be mutually exclusive";

    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || !dstSpace)
      return emitOpError() << "expects src and dst to have explicit address spaces";

    auto srcShape = getShapeVec(srcTy);
    auto dstShape = getShapeVec(dstTy);
    if (*srcSpace == pto::AddressSpace::MAT && srcShape != dstShape)
      return emitOpError() << "expects mat-source tmov to use matching src/dst shapes";
    if (!isA5 && *srcSpace != pto::AddressSpace::MAT && srcShape != dstShape)
      return emitOpError() << "expects A2/A3 non-mat tmov to use matching src/dst shapes";

    const bool isMatToTile =
        *srcSpace == pto::AddressSpace::MAT &&
        (*dstSpace == pto::AddressSpace::LEFT ||
         *dstSpace == pto::AddressSpace::RIGHT ||
         *dstSpace == pto::AddressSpace::BIAS ||
         *dstSpace == pto::AddressSpace::SCALING);
    const bool isVecToVec =
        *srcSpace == pto::AddressSpace::VEC &&
        *dstSpace == pto::AddressSpace::VEC;
    const bool isVecToMat =
        *srcSpace == pto::AddressSpace::VEC &&
        *dstSpace == pto::AddressSpace::MAT;
    const bool isAccToMat =
        *srcSpace == pto::AddressSpace::ACC &&
        *dstSpace == pto::AddressSpace::MAT;
    const bool isAccToVec =
        *srcSpace == pto::AddressSpace::ACC &&
        *dstSpace == pto::AddressSpace::VEC;

    bool okPair = isMatToTile || isVecToVec || isAccToMat || isAccToVec;
    if (isA5)
      okPair = okPair || isVecToMat;
    if (!okPair)
      return emitOpError()
             << "expects a supported tmov address-space pair for this target";

    if (accToVecModeAttr && !isAccToVec)
      return emitOpError()
             << "expects accToVecMode to be used only for acc-to-vec tmov";

    if (reluMode != pto::ReluPreMode::NoRelu && !(isAccToMat || isAccToVec))
      return emitOpError()
             << "expects reluPreMode form to use loc=acc src";

    if (hasPreQuantScalar && !(isAccToMat || isAccToVec))
      return emitOpError()
             << "expects preQuantScalar form to use loc=acc src";

    if (hasFp) {
      auto fpTy = fp.getType();
      auto fpSpace = getPTOMemorySpaceEnum(fpTy);
      if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING)
        return emitOpError() << "expects fp to be in the scaling address space";
      auto srcElemTy = getElemTy(srcTy);
      auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
      if (!(srcElemTy.isF32() || (srcIntTy && srcIntTy.getWidth() == 32)))
        return emitOpError()
               << "expects fp form src to have element type f32, i32";
      if (!(isAccToMat || isAccToVec))
        return emitOpError() << "expects fp form to use loc=acc src";
    }

    if ((hasFp || hasPreQuantScalar) && accToVecModeAttr) {
      switch (accToVecModeAttr.getValue()) {
      case pto::AccToVecMode::SingleModeVec0:
      case pto::AccToVecMode::SingleModeVec1:
        break;
      case pto::AccToVecMode::DualModeSplitM:
      case pto::AccToVecMode::DualModeSplitN:
        return emitOpError()
               << "expects fp/preQuantScalar acc-to-vec forms to use single-mode accToVecMode";
      }
    }

    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (srcTb && *srcSpace == pto::AddressSpace::ACC &&
        (hasFp || reluMode != pto::ReluPreMode::NoRelu)) {
      if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
          srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))
        return emitOpError()
               << "expects acc-source fp/relu tmov src to use blayout=col_major and slayout=row_major";
    }
    if (srcTb && dstTb && isAccToMat && !isA5 &&
        dstTb.getSFractalSizeI32() != 512)
      return emitOpError() << "expects A2/A3 acc-to-mat tmov destination fractal to be 512";

    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyImpl(/*isA5=*/false); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyImpl(/*isA5=*/true); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMovFPOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type fpTy  = getFp().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto srcElemTy = getElemTy(srcTy);
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() ||
          (srcIntTy && srcIntTy.getWidth() == 32)))
      return emitOpError()
             << "expects src to have element type f32, i32";
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != mlir::pto::AddressSpace::SCALING)
      return emitOpError() << "expects fp to be in the scaling address space";
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace || *srcSpace != mlir::pto::AddressSpace::ACC)
      return emitOpError() << "expects src to be in the acc address space";
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || *dstSpace != mlir::pto::AddressSpace::MAT)
      return emitOpError() << "expects dst to be in the mat address space";
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
    if (srcTb &&
        (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
         srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)))
      return emitOpError()
             << "expects src to use blayout=col_major and slayout=row_major";
    if (dstTb &&
        (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
         dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)))
      return emitOpError()
             << "expects dst to use blayout=col_major and slayout=row_major";
    if (dstTb && dstTb.getSFractalSizeI32() != 512)
      return emitOpError() << "expects dst to use fractal size 512";
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type fpTy  = getFp().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    auto srcElemTy = getElemTy(srcTy);
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() ||
          (srcIntTy && srcIntTy.getWidth() == 32)))
      return emitOpError()
             << "expects src to have element type f32, i32";
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != mlir::pto::AddressSpace::SCALING)
      return emitOpError() << "expects fp to be in the scaling address space";
    auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
    if (srcTb &&
        (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
         srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)))
      return emitOpError()
             << "expects src to use blayout=col_major and slayout=row_major";
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
// 辅助函数：获取 Rank，支持 ShapedType 和 PTO TileTypes
static int64_t getRankHelper(Type t) {
  if (auto s = dyn_cast<ShapedType>(t)) return s.getRank();
  if (auto tile = dyn_cast<pto::TileBufType>(t)) return tile.getRank();
  if (auto view = dyn_cast<pto::PartitionTensorViewType>(t)) return view.getRank();
  return -1;
}

static LogicalResult verifyMatmulLike(Operation *op, Type aTy, Type bTy, Type dstTy, bool checkRank = true) {
  // 1. 检查类型 (ShapedType 或 Tile 类型)
  bool aValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(aTy);
  bool bValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(bTy);
  bool dValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(dstTy);

  if (!aValid || !bValid || !dValid)
    return op->emitOpError("expects inputs/outputs to be shaped types or PTO tile types");

  if (checkRank) {
    int64_t aRank = getRankHelper(aTy);
    int64_t bRank = getRankHelper(bTy);
    int64_t dRank = getRankHelper(dstTy);

    // 检查 Rank 一致性
    if (aRank != -1 && dRank != -1 && aRank != dRank)
      return op->emitOpError("expects a and dst to have the same rank");
    if (bRank != -1 && dRank != -1 && bRank != dRank)
      return op->emitOpError("expects b and dst to have the same rank");
  }

  return success();
}

// ---- LoadScalarOp ----
LogicalResult LoadScalarOp::verify() {
  Type ptrTy = getPtr().getType();
  Type elemTy;
  if (auto pty = dyn_cast<mlir::pto::PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(ptrTy)) {
    elemTy = memTy.getElementType();
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError() << "scalar load only supports GM address space pointers";
  } else {
    return emitOpError("expects ptr to be !pto.ptr or memref type");
  }

  if (getValue().getType() != elemTy)
    return emitOpError("expects result type to match ptr element type");

  return success();
}
// ---- StoreScalarOp ----
LogicalResult StoreScalarOp::verify() {
  Type ptrTy = getPtr().getType();
  Type elemTy;
  if (auto pty = dyn_cast<mlir::pto::PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(ptrTy)) {
    elemTy = memTy.getElementType();
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError() << "scalar store only supports GM address space pointers";
  } else {
    return emitOpError("expects ptr to be !pto.ptr or memref type");
  }

  if (getValue().getType() != elemTy)
    return emitOpError("expects value type to match ptr element type");

  return success();
}

// ---- GetBufOp / RlsBufOp ----
static LogicalResult verifyBufSyncOp(Operation *op, Attribute opTypeAttr,
                                     IntegerAttr bufIdAttr, IntegerAttr modeAttr) {
  if (!opTypeAttr)
    return op->emitOpError("expects 'op_type' attribute");

  pto::PIPE pipe = pto::PIPE::PIPE_UNASSIGNED;
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    pipe = pipeAttr.getPipe();
  } else {
    auto opTypeOr = parseSyncOpTypeLikeAttr(opTypeAttr);
    if (failed(opTypeOr)) {
      auto diag = op->emitOpError(
          "expects 'op_type' to be pipe_event_type/sync_op_type/pipe, got ");
      diag << opTypeAttr;
      return failure();
    }
    pipe = mapSyncOpTypeToPipe(*opTypeOr);
  }
  if (!isConcreteSyncPipe(pipe))
    return op->emitOpError("expects 'op_type' to map to a concrete pipe, not PIPE_ALL/PIPE_UNASSIGNED");

  if (!bufIdAttr)
    return op->emitOpError("expects 'buf_id' attribute");
  int64_t bufId = bufIdAttr.getInt();
  if (bufId < 0 || bufId > 31)
    return op->emitOpError("expects 'buf_id' in range [0, 31]");

  if (modeAttr) {
    int64_t mode = modeAttr.getInt();
    if (mode < 0)
      return op->emitOpError("expects 'mode' to be non-negative");
  }

  return success();
}

LogicalResult GetBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

LogicalResult RlsBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

static ParseResult parseLegacyOrAttrMemBar(OpAsmParser &parser,
                                           MemBarAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeMemBarKind(token);
    if (!kind)
      return parser.emitError(loc) << "invalid membar token: " << token;
    attr = MemBarAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed)))
    return failure();
  auto memBarAttr = dyn_cast<MemBarAttr>(parsed);
  if (!memBarAttr)
    return parser.emitError(loc, "expected membar attribute");
  attr = memBarAttr;
  return success();
}

static void printLegacyOrAttrMemBar(OpAsmPrinter &p, MemBarAttr kind,
                                    ArrayRef<NamedAttribute> attrs) {
  p << ' ' << '"' << stringifyMemBarKind(kind.getKind()) << '"';
  p.printOptionalAttrDict(attrs, {"kind"});
}

static ParseResult parseLegacyOrAttrPipe(OpAsmParser &parser, PipeAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto pipe = symbolizePIPE(token);
    if (!pipe)
      return parser.emitError(loc) << "invalid pipe token: " << token;
    attr = PipeAttr::get(parser.getContext(), *pipe);
    return success();
  }

  if (succeeded(parser.parseOptionalLess())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseGreater())
      return failure();
    auto pipe = symbolizePIPE(keyword);
    if (!pipe)
      return parser.emitError(loc) << "invalid pipe token: " << keyword;
    attr = PipeAttr::get(parser.getContext(), *pipe);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed)))
    return failure();
  auto pipeAttr = dyn_cast<PipeAttr>(parsed);
  if (!pipeAttr)
    return parser.emitError(loc, "expected pipe attribute");
  attr = pipeAttr;
  return success();
}

static ParseResult parseLegacyOrAttrEvent(OpAsmParser &parser, EventAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto event = symbolizeEVENT(token);
    if (!event)
      return parser.emitError(loc) << "invalid event token: " << token;
    attr = EventAttr::get(parser.getContext(), *event);
    return success();
  }

  if (succeeded(parser.parseOptionalLess())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseGreater())
      return failure();
    auto event = symbolizeEVENT(keyword);
    if (!event)
      return parser.emitError(loc) << "invalid event token: " << keyword;
    attr = EventAttr::get(parser.getContext(), *event);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed)))
    return failure();
  auto eventAttr = dyn_cast<EventAttr>(parsed);
  if (!eventAttr)
    return parser.emitError(loc, "expected event attribute");
  attr = eventAttr;
  return success();
}

static ParseResult parseI32LiteralAttr(OpAsmParser &parser, IntegerAttr &attr) {
  auto loc = parser.getCurrentLocation();
  int64_t value = 0;
  if (failed(parser.parseInteger(value)))
    return failure();
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max())
    return parser.emitError(loc, "expected 32-bit integer literal");
  attr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), value);
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

ParseResult SetFlagOp::parse(OpAsmParser &parser, OperationState &result) {
  PipeAttr srcPipe;
  PipeAttr dstPipe;
  EventAttr eventId;
  if (parser.parseLSquare() || parseLegacyOrAttrPipe(parser, srcPipe) ||
      parser.parseComma() || parseLegacyOrAttrPipe(parser, dstPipe) ||
      parser.parseComma() || parseLegacyOrAttrEvent(parser, eventId) ||
      parser.parseRSquare())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("src_pipe", srcPipe);
  result.addAttribute("dst_pipe", dstPipe);
  result.addAttribute("event_id", eventId);
  return success();
}

void SetFlagOp::print(OpAsmPrinter &p) {
  printLegacySyncTriplet(p, getSrcPipe(), getDstPipe(), getEventId(),
                         (*this)->getAttrs());
}

ParseResult WaitFlagOp::parse(OpAsmParser &parser, OperationState &result) {
  PipeAttr srcPipe;
  PipeAttr dstPipe;
  EventAttr eventId;
  if (parser.parseLSquare() || parseLegacyOrAttrPipe(parser, srcPipe) ||
      parser.parseComma() || parseLegacyOrAttrPipe(parser, dstPipe) ||
      parser.parseComma() || parseLegacyOrAttrEvent(parser, eventId) ||
      parser.parseRSquare())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("src_pipe", srcPipe);
  result.addAttribute("dst_pipe", dstPipe);
  result.addAttribute("event_id", eventId);
  return success();
}

void WaitFlagOp::print(OpAsmPrinter &p) {
  printLegacySyncTriplet(p, getSrcPipe(), getDstPipe(), getEventId(),
                         (*this)->getAttrs());
}

ParseResult MemBarOp::parse(OpAsmParser &parser, OperationState &result) {
  MemBarAttr kind;
  if (parseLegacyOrAttrMemBar(parser, kind))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("kind", kind);
  return success();
}

void MemBarOp::print(OpAsmPrinter &p) {
  printLegacyOrAttrMemBar(p, getKind(), (*this)->getAttrs());
}

static ParseResult parseBufSyncOp(OpAsmParser &parser, OperationState &result) {
  Attribute opTypeAttr;
  IntegerAttr bufIdAttr;
  IntegerAttr modeAttr;

  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    if (auto pipe = symbolizePIPE(token))
      opTypeAttr = PipeAttr::get(parser.getContext(), *pipe);
    else if (auto opType = symbolizeSyncOpType(token))
      opTypeAttr = PipeEventTypeAttr::get(parser.getContext(), *opType);
    else
      return parser.emitError(loc) << "invalid get_buf/rls_buf token: " << token;

    if (parser.parseComma() || parseI32LiteralAttr(parser, bufIdAttr))
      return failure();
    if (succeeded(parser.parseOptionalComma())) {
      if (parseI32LiteralAttr(parser, modeAttr))
        return failure();
    } else {
      modeAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), 0);
    }
  } else if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseAttribute(opTypeAttr) || parser.parseComma() ||
        parseI32LiteralAttr(parser, bufIdAttr))
      return failure();
    if (succeeded(parser.parseOptionalComma())) {
      if (parseI32LiteralAttr(parser, modeAttr))
        return failure();
    } else {
      modeAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), 0);
    }
    if (parser.parseRSquare())
      return failure();
  } else {
    return parser.emitError(loc, "expected string pipe/op_type or '['");
  }

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("op_type", opTypeAttr);
  result.addAttribute("buf_id", bufIdAttr);
  result.addAttribute("mode", modeAttr);
  return success();
}

static void printBufSyncOp(OpAsmPrinter &p, Attribute opTypeAttr,
                           IntegerAttr bufIdAttr, IntegerAttr modeAttr,
                           ArrayRef<NamedAttribute> attrs) {
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    p << " \"" << stringifyPIPE(pipeAttr.getPipe()) << "\", "
      << bufIdAttr.getInt() << ", " << modeAttr.getInt();
  } else if (auto pipeEventType = dyn_cast<PipeEventTypeAttr>(opTypeAttr)) {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  } else if (auto syncOpType = dyn_cast<SyncOpTypeAttr>(opTypeAttr)) {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  } else {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  }
  p.printOptionalAttrDict(attrs, {"op_type", "buf_id", "mode"});
}

ParseResult GetBufOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufSyncOp(parser, result);
}

void GetBufOp::print(OpAsmPrinter &p) {
  printBufSyncOp(p, getOpTypeAttr(), getBufIdAttr(), getModeAttr(),
                 (*this)->getAttrs());
}

ParseResult RlsBufOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufSyncOp(parser, result);
}

void RlsBufOp::print(OpAsmPrinter &p) {
  printBufSyncOp(p, getOpTypeAttr(), getBufIdAttr(), getModeAttr(),
                 (*this)->getAttrs());
}
// ---- TOp ----
LogicalResult TGemvBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyGemvTileOperands(*this, getA().getType(), getB().getType(),
                                      getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType())))
      return failure();
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getA().getType()),
                                      getElemTy(getB().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyScaleTileMatchesOperand(*this, getAScale().getType(),
                                             getA().getType(), "a_scale", "a")) ||
        failed(verifyScaleTileMatchesOperand(*this, getBScale().getType(),
                                             getB().getType(), "b_scale", "b")) ||
        failed(verifyA5MxGemvTileOperands(*this, getA().getType(), getB().getType(),
                                          getDst().getType())))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx.acc is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getCIn().getType(), "c_in")) ||
        failed(verifyScaleTileMatchesOperand(*this, getAScale().getType(),
                                             getA().getType(), "a_scale", "a")) ||
        failed(verifyScaleTileMatchesOperand(*this, getBScale().getType(),
                                             getB().getType(), "b_scale", "b")) ||
        failed(verifyA5MxGemvTileOperands(*this, getA().getType(), getB().getType(),
                                          getDst().getType())))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, getCIn().getType(),
                                             getDst().getType(), "c_in", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, getCIn().getType(),
                                           getDst().getType(), "c_in", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx.bias is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyScaleTileMatchesOperand(*this, getAScale().getType(),
                                             getA().getType(), "a_scale", "a")) ||
        failed(verifyScaleTileMatchesOperand(*this, getBScale().getType(),
                                             getB().getType(), "b_scale", "b")) ||
        failed(verifyGemvTileOperands(*this, getA().getType(), getB().getType(),
                                      getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType(),
                                 /*requireFloatBias=*/true)))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst")))
      return failure();
    auto biasShape = getShapeVec(getBias().getType());
    auto dstShape = getShapeVec(getDst().getType());
    if (biasShape.size() != 2 || dstShape.size() != 2)
      return emitOpError("expects bias and dst to be rank-2 for tgemv.mx.bias");
    if (biasShape[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        biasShape[1] != dstShape[1])
      return emitOpError("expects bias and dst to have the same column shape");
    if (failed(verifyTileBufSameValidShape(*this, getBias().getType(),
                                           getDst().getType(), "bias", "dst")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyMatTileOperands(*this, getA().getType(), getB().getType(),
                                         getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType())))
      return failure();
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getA().getType()),
                                      getElemTy(getB().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyTileBufCommon(*this, getAScale().getType(), "a_scale")) ||
        failed(verifyTileBufCommon(*this, getBScale().getType(), "b_scale")))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA2A3()))
      return failure();
    return verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                getDst().getType(), "lhs", "rhs", "dst");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulMxAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getCIn().getType(), "c_in")) ||
        failed(verifyTileBufCommon(*this, getAScale().getType(), "a_scale")) ||
        failed(verifyTileBufCommon(*this, getBScale().getType(), "b_scale")))
      return failure();
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA2A3()))
      return failure();
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, getCIn().getType(),
                                             getDst().getType(), "c_in", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, getCIn().getType(),
                                           getDst().getType(), "c_in", "dst")))
      return failure();
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult TMatmulMxBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyTileBufCommon(*this, getAScale().getType(), "a_scale")) ||
        failed(verifyTileBufCommon(*this, getBScale().getType(), "b_scale")) ||
        failed(verifyMatTileOperands(*this, getA().getType(), getB().getType(),
                                         getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType(),
                              /*requireFloatBias=*/true)))
      return failure();
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA2A3()))
      return failure();
    return verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                getDst().getType(), "lhs", "rhs", "dst");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
// ---- TSetValOp ----
LogicalResult TSetValOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  // dst can be tile/tensor/tilebuf (PTODpsType). Keep checks minimal.
  if (auto shaped = dyn_cast<ShapedType>(getDst().getType())) {
    if (shaped.getElementType() != getVal().getType())
      return emitOpError("expects val type to match dst element type");
  }
  return success();
}
// ---- TGetValOp ----
LogicalResult TGetValOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  if (!mlir::isa<pto::TileBufType, MemRefType>(srcTy))
    return emitOpError("expects src to be tile_buf or memref type");

  // Memory space must be vec (Ascend does not support getval from MAT etc.).
  Attribute memSpace =
      isa<pto::TileBufType>(srcTy)
          ? cast<pto::TileBufType>(srcTy).getMemorySpace()
          : cast<MemRefType>(srcTy).getMemorySpace();
  auto addrSpaceAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memSpace);
  if (!addrSpaceAttr ||
      addrSpaceAttr.getAddressSpace() != pto::AddressSpace::VEC) {
    if (addrSpaceAttr &&
        addrSpaceAttr.getAddressSpace() == pto::AddressSpace::MAT)
      return emitOpError(
          "Ascend hardware does not support reading from Mat tile_buf to Scalar unit");
    return emitOpError("expects src memory space to be vec");
  }

  if (getElemTy(srcTy) != getDst().getType())
    return emitOpError("expects dst type to match src element type");
  return success();
}

LogicalResult THistogramOp::verify() {
  auto isIntegerWidth = [](Type ty, unsigned width) {
    auto it = dyn_cast<IntegerType>(ty);
    return it && it.getWidth() == width;
  };
  int64_t byte = 1;
  auto byteAttr = getByteAttr();
  if (byteAttr)
    byte = byteAttr.getInt();
  if (auto legacyIsMSB = (*this)->getAttrOfType<BoolAttr>("isMSB")) {
    int64_t legacyByte = legacyIsMSB.getValue() ? 1 : 0;
    if (byteAttr && byte != legacyByte)
      return emitOpError("does not allow conflicting 'byte' and legacy 'isMSB' attributes");
    byte = legacyByte;
  }
  if (byte < 0 || byte > 3)
    return emitOpError("expects byte to be in range [0, 3]");

  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("thistogram is only supported on A5");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type idxTy = getIdx().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, idxTy, "idx")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();

    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto idxSpace = getPTOMemorySpaceEnum(idxTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::VEC)
      return emitOpError("expects src to be in the vec address space");
    if (!idxSpace || *idxSpace != pto::AddressSpace::VEC)
      return emitOpError("expects idx to be in the vec address space");
    if (!dstSpace || *dstSpace != pto::AddressSpace::VEC)
      return emitOpError("expects dst to be in the vec address space");

    auto srcTB = dyn_cast<pto::TileBufType>(srcTy);
    auto idxTB = dyn_cast<pto::TileBufType>(idxTy);
    auto dstTB = dyn_cast<pto::TileBufType>(dstTy);
    if (!srcTB || !idxTB || !dstTB)
      return emitOpError("expects src, idx, and dst to be tile_buf types");

    if (srcTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        srcTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
      return emitOpError("expects src to use row_major + none_box layout");
    if (dstTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        dstTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
      return emitOpError("expects dst to use row_major + none_box layout");

    bool srcIsUi16 = isIntegerWidth(getElemTy(srcTy), 16);
    bool srcIsUi32 = isIntegerWidth(getElemTy(srcTy), 32);
    if (!srcIsUi16 && !srcIsUi32)
      return emitOpError("expects src element type to be ui16 or ui32");
    if (!isIntegerWidth(getElemTy(idxTy), 8))
      return emitOpError("expects idx element type to be ui8");
    if (!isIntegerWidth(getElemTy(dstTy), 32))
      return emitOpError("expects dst element type to be ui32");

    auto srcShape = getShapeVec(srcTy);
    auto idxShape = getShapeVec(idxTy);
    auto dstShape = getShapeVec(dstTy);
    auto srcValid = getValidShapeVec(srcTy);
    auto idxValid = getValidShapeVec(idxTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcShape.size() != 2 || idxShape.size() != 2 || dstShape.size() != 2 ||
        srcValid.size() != 2 || idxValid.size() != 2 || dstValid.size() != 2)
      return emitOpError(
          "expects src, idx, and dst to have rank-2 shape and valid_shape");

    if (!hasCompatibleKnownExtent(srcShape[0], dstShape[0]) ||
        !hasCompatibleKnownExtent(srcValid[0], dstValid[0]))
      return emitOpError("expects dst rows and valid rows to match src");

    if (srcIsUi16) {
      if (byte > 1)
        return emitOpError("expects byte to be 0 or 1 when src element type is ui16");
      if (idxTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
          idxTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
        return emitOpError(
            "expects idx to use DN layout (col_major + none_box) when src element type is ui16");
      if (!hasCompatibleKnownExtent(srcShape[0], idxShape[0]) ||
          !hasCompatibleKnownExtent(srcValid[0], idxValid[0]))
        return emitOpError("expects idx rows and valid rows to match src when src element type is ui16");
      if (!isKnownUnitExtent(idxShape[1]) || !isKnownZeroOrUnitExtent(idxValid[1]))
        return emitOpError("expects idx to have exactly one physical column and 0 or 1 valid column when src element type is ui16");
    } else {
      if (byte != 3) {
        if (idxTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
            idxTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
          return emitOpError(
              "expects idx to use row_major + none_box layout when src element type is ui32 and byte is 0, 1, or 2");
        if (!hasCompatibleKnownExtent(srcShape[1], idxShape[1]) ||
            !hasCompatibleKnownExtent(srcValid[1], idxValid[1]))
          return emitOpError(
              "expects idx cols and valid cols to match src when src element type is ui32 and byte is 0, 1, or 2");

        int64_t expectedIdxRows = 1;
        if (byte == 1)
          expectedIdxRows = 2;
        else if (byte == 0)
          expectedIdxRows = 3;
        if (!hasCompatibleKnownExtent(idxShape[0], expectedIdxRows) ||
            !hasCompatibleKnownExtentOrZero(idxValid[0], expectedIdxRows))
          return emitOpError(
              "expects idx rows to match the byte-selected filter depth and idx valid rows to be 0 or match it when src element type is ui32 and byte is 0, 1, or 2");
      }
    }
    if (dstShape[1] != ShapedType::kDynamic && dstShape[1] < 256)
      return emitOpError("expects dst shape[1] to be at least 256");
    if (dstValid[1] != ShapedType::kDynamic && dstValid[1] != 0 &&
        dstValid[1] < 256)
      return emitOpError("expects dst valid_shape[1] to be 0 or at least 256");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGetScaleAddrOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tget_scale_addr is only supported on A5");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")))
      return failure();
    if (failed(verifyScaleTileMatchesOperand(*this, dstTy, srcTy, "dst", "src")))
      return failure();
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- MScatterOp ----
LogicalResult MScatterOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  if (!isTargetArchA5(getOperation()))
    return emitOpError("pto.mscatter is only supported on A5 targets");

  Type srcTy = getSrc().getType();
  Type idxTy = getIdx().getType();
  Type memTy = getMem().getType();

  if (getPTOTypeRank(srcTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(memTy) == -1)
    return emitOpError("expects src, idx, and mem to use supported PTO shapes");

  if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
      failed(verifyMGatherMScatterIdxTile(getOperation(), idxTy, "idx")))
    return failure();

  Type srcElem = getElemTy(srcTy);
  Type idxElem = getElemTy(idxTy);
  if (!srcElem || !idxElem)
    return emitOpError("failed to resolve element types for src or idx");

  if (!isSupportedMGatherMScatterPayloadElemType(getOperation(), srcElem))
    return emitOpError(
        "expects src element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");

  if (!isSupportedMGatherMScatterIndexElemType(idxElem))
    return emitOpError("expects idx element type to be signless i32");

  if (failed(verifyMGatherMScatterMemOperand(getOperation(), getMem(), srcElem,
                                             "src")))
    return failure();

  if (getScatterAtomicOp() != pto::ScatterAtomicOp::None ||
      getScatterOob() != pto::ScatterOOB::Undefined) {
    if (!isTargetArchA5(getOperation()))
      return emitOpError(
          "expects non-default scatterAtomicOp/scatterOob only on A5 targets");
  }

  if (!isSupportedMScatterAtomicPayloadElemType(srcElem, getScatterAtomicOp()))
    return emitOpError(
        "expects scatterAtomicOp-compatible src element type: add supports "
        "i32/ui32/f16/f32, max/min support signless i32/f32");

  if (failed(verifyMGatherMScatterTileShape(getOperation(), srcTy, idxTy, "src")))
    return failure();

  return success();
}

// ---- MGatherOp ----
LogicalResult MGatherOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  if (!isTargetArchA5(getOperation()))
    return emitOpError("pto.mgather is only supported on A5 targets");

  Type memTy = getMem().getType();
  Type idxTy = getIdx().getType();
  Type dstTy = getDst().getType();

  if (getPTOTypeRank(memTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(dstTy) == -1)
    return emitOpError("expects mem, idx, and dst to use supported PTO shapes");

  if (failed(verifyNDStyleVecTile(*this, dstTy, "dst")) ||
      failed(verifyMGatherMScatterIdxTile(getOperation(), idxTy, "idx")))
    return failure();

  Type dstElem = getElemTy(dstTy);
  Type idxElem = getElemTy(idxTy);
  if (!dstElem || !idxElem)
    return emitOpError("failed to resolve element types for dst or idx");

  if (!isSupportedMGatherMScatterPayloadElemType(getOperation(), dstElem))
    return emitOpError(
        "expects dst element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");

  if (!isSupportedMGatherMScatterIndexElemType(idxElem))
    return emitOpError("expects idx element type to be signless i32");

  if (failed(verifyMGatherMScatterMemOperand(getOperation(), getMem(), dstElem,
                                             "dst")))
    return failure();

  if (getGatherOob() != pto::GatherOOB::Undefined &&
      !isTargetArchA5(getOperation()))
    return emitOpError(
        "expects non-default gatherOob only on A5 targets");

  if (failed(verifyMGatherMScatterTileShape(getOperation(), dstTy, idxTy, "dst")))
    return failure();

  return success();
}

void mlir::pto::TCvtOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  Builder builder(getContext());
  NamedAttrList attrs;
  for (auto attr : (*this)->getAttrs()) {
    if (attr.getName() == "sat_mode") {
      attrs.set(builder.getStringAttr("satmode"), attr.getValue());
      continue;
    }
    attrs.set(attr.getName(), attr.getValue());
  }
  p.printOptionalAttrDict(attrs.getAttrs());
  p << " : " << getSrc().getType();
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
}

ParseResult mlir::pto::TCvtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, dst;
  Type srcTy, dstTy;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs) || parser.parseColonType(srcTy))
    return failure();
  if (auto satmode = attrs.get("satmode")) {
    attrs.erase("satmode");
    if (attrs.get("sat_mode"))
      return parser.emitError(parser.getCurrentLocation(),
                              "cannot specify both satmode and sat_mode");
    attrs.set("sat_mode", satmode);
  }
  result.attributes = attrs;
  if (parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) || parser.parseRParen())
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands))
    return failure();
  return success();
}

void mlir::pto::TMrgSortOp::print(OpAsmPrinter &p) {
  if (isFormat1()) {
    p << " ins(" << getSrc() << ", " << getBlockLen() << " : " << getSrc().getType()
      << ", " << getBlockLen().getType() << ") outs(" << getDst() << " : "
      << getDst().getType() << ")";
  } else if (isFormat2()) {
    p << " ins(";
    llvm::interleaveComma(getSrcs(), p, [&](Value src) { p << src; });
    p << ", " << getTmp();
    p << " {exhausted = " << (getExhausted() ? "true" : "false") << "} : ";
    llvm::interleaveComma(getSrcs().getTypes(), p, [&](Type ty) { p << ty; });
    p << ", " << getTmp().getType();
    p << ") outs(" << getDst() << ", " << getExcuted()
      << " : " << getDst().getType() << ", " << getExcuted().getType() << ")";
  } else {
    llvm::report_fatal_error("TMrgSortOp print expects format1 or format2");
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes", "exhausted"});
}

ParseResult mlir::pto::TMrgSortOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseKeyword("ins") || parser.parseLParen())
    return failure();
  OpAsmParser::UnresolvedOperand first, second;
  if (parser.parseOperand(first) || parser.parseComma() || parser.parseOperand(second))
    return failure();

  if (parser.parseOptionalColon().succeeded()) {
    Type srcTy, blockLenTy, dstTy;
    if (parser.parseType(srcTy) || parser.parseComma() || parser.parseType(blockLenTy) ||
        parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen())
      return failure();
    OpAsmParser::UnresolvedOperand dstOp;
    if (parser.parseOperand(dstOp) || parser.parseColon() || parser.parseType(dstTy) ||
        parser.parseRParen())
      return failure();
    result.addAttribute("operandSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr({1, 1, 1, 0, 0}));
    if (parser.resolveOperand(first, srcTy, result.operands) ||
        parser.resolveOperand(second, blockLenTy, result.operands) ||
        parser.resolveOperand(dstOp, dstTy, result.operands))
      return failure();
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();
    if (!result.attributes.get("exhausted"))
      result.addAttribute("exhausted", parser.getBuilder().getBoolAttr(false));
    return success();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 4> srcs = {first, second};
  while (parser.parseOptionalComma().succeeded()) {
    OpAsmParser::UnresolvedOperand next;
    if (parser.parseOperand(next))
      return failure();
    srcs.push_back(next);
  }
  if (srcs.size() < 3 || srcs.size() > 5)
    return parser.emitError(parser.getCurrentLocation(),
                            "tmrgsort format2 expects 2 to 4 src operands plus one tmp operand");
  OpAsmParser::UnresolvedOperand tmpOp = srcs.pop_back_val();
  bool exhaustedVal = false;
  if (parser.parseOptionalLBrace().succeeded()) {
    if (parser.parseKeyword("exhausted") || parser.parseEqual())
      return failure();
    StringRef kw;
    if (parser.parseKeyword(&kw) || parser.parseRBrace())
      return failure();
    exhaustedVal = (kw == "true");
  }
  SmallVector<Type, 4> srcTypes;
  srcTypes.reserve(srcs.size());
  if (parser.parseColon())
    return failure();
  Type firstSrcTy;
  if (parser.parseType(firstSrcTy))
    return failure();
  srcTypes.push_back(firstSrcTy);
  while (parser.parseOptionalComma().succeeded()) {
    Type nextTy;
    if (parser.parseType(nextTy))
      return failure();
    srcTypes.push_back(nextTy);
  }
  if (srcTypes.size() != srcs.size() + 1 || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen())
    return failure();
  Type tmpTy = srcTypes.pop_back_val();
  OpAsmParser::UnresolvedOperand dstOp, excutedOp;
  Type dstTy, excutedTy;
  if (parser.parseOperand(dstOp) || parser.parseComma() || parser.parseOperand(excutedOp) ||
      parser.parseColon() || parser.parseType(dstTy) || parser.parseComma() ||
      parser.parseType(excutedTy) || parser.parseRParen())
    return failure();
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(srcs.size()), 0, 1, 1, 1}));
  if (parser.resolveOperands(srcs, srcTypes, parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperand(dstOp, dstTy, result.operands) ||
      parser.resolveOperand(tmpOp, tmpTy, result.operands) ||
      parser.resolveOperand(excutedOp, excutedTy, result.operands))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (!result.attributes.get("exhausted"))
    result.addAttribute("exhausted", parser.getBuilder().getBoolAttr(exhaustedVal));
  return success();
}

mlir::LogicalResult mlir::pto::TMrgSortOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (isFormat1()) {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy))
      return emitOpError() << "format1 expects PTO shaped-like types for src/dst";
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError() << "expects src/dst to have the same element type";
    if (!getElemTy(srcTy).isF16() && !getElemTy(srcTy).isF32())
      return emitOpError() << "expects element type to be f16 or f32";
    auto ss = getShapeVec(srcTy);
    auto ds = getShapeVec(dstTy);
    if (ss.size() != 2 || ds.size() != 2)
      return emitOpError() << "expects src/dst to be rank-2 tile-shaped";
    if (ss[0] != mlir::ShapedType::kDynamic && ss[0] != 1)
      return emitOpError() << "expects src rows == 1";
    if (ds[0] != mlir::ShapedType::kDynamic && ds[0] != 1)
      return emitOpError() << "expects dst rows == 1";
    if (ss[1] != mlir::ShapedType::kDynamic && ds[1] != mlir::ShapedType::kDynamic && ss[1] != ds[1])
      return emitOpError() << "expects src/dst cols to match";
    if (getBlockLen()) {
      if (auto cstOp = getBlockLen().getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cstOp.getValue())) {
          int64_t v = intAttr.getValue().getSExtValue();
          if (v <= 0 || (v % 64) != 0)
            return emitOpError() << "expects blockLen > 0 and multiple of 64";
        }
      }
    }
    return mlir::success();
  }
  if (isFormat2()) {
    for (Value v : getSrcs())
      if (!isPTOShapedLike(v.getType()))
        return emitOpError() << "format2 expects PTO shaped-like type for each src";
    if (getSrcs().size() < 2u || getSrcs().size() > 4u)
      return emitOpError() << "format2 expects 2 to 4 srcs";
    if (getDsts().size() != 1u || !getTmp() || !getExcuted())
      return emitOpError() << "format2 expects ins(srcs..., tmp), outs(dst), and excuted=vector";
    Type dstTy = getDst().getType();
    Type tmpTy = getTmp().getType();
    if (!isPTOShapedLike(dstTy) || !isPTOShapedLike(tmpTy))
      return emitOpError() << "format2 dst/tmp must be PTO shaped-like";
    auto excutedTy = mlir::dyn_cast<mlir::VectorType>(getExcuted().getType());
    if (!excutedTy || excutedTy.getRank() != 1 || excutedTy.getNumElements() != 4 ||
        !excutedTy.getElementType().isInteger(16))
      return emitOpError() << "format2 excuted must be vector<4xi16>";
    Type elemTy = getElemTy(dstTy);
    if (elemTy != getElemTy(tmpTy))
      return emitOpError() << "format2 expects dst/tmp element types to match";
    auto dstShape = getShapeVec(dstTy);
    auto tmpShape = getShapeVec(tmpTy);
    if (dstShape.size() != 2 || tmpShape.size() != 2)
      return emitOpError() << "format2 expects dst/tmp to be rank-2 tile-shaped";
    if ((dstShape[0] != mlir::ShapedType::kDynamic && dstShape[0] != 1) ||
        (tmpShape[0] != mlir::ShapedType::kDynamic && tmpShape[0] != 1))
      return emitOpError() << "format2 expects dst/tmp rows == 1";
    if (dstShape[1] != mlir::ShapedType::kDynamic &&
        tmpShape[1] != mlir::ShapedType::kDynamic &&
        tmpShape[1] < dstShape[1])
      return emitOpError() << "format2 expects tmp.cols >= dst.cols";
    for (Value src : getSrcs()) {
      Type srcTy = src.getType();
      auto srcShape = getShapeVec(srcTy);
      if (srcShape.size() != 2)
        return emitOpError() << "format2 expects src to be rank-2 tile-shaped";
      if (srcShape[0] != mlir::ShapedType::kDynamic && srcShape[0] != 1)
        return emitOpError() << "format2 expects src rows == 1";
      if (getElemTy(srcTy) != elemTy)
        return emitOpError() << "format2 expects src/dst/tmp element types to match";
    }
    return mlir::success();
  }
  return emitOpError() << "tmrgsort expects format1 (1 src + blockLen + 1 dst) or "
                          "format2 (2 to 4 srcs + tmp, outs dst, excuted)";
}

mlir::LogicalResult mlir::pto::TMulOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tmul element type to be i32/i16/f16/f32",
      "expects A5 tmul element type to be i32/i16/f16/f32");
}

mlir::LogicalResult mlir::pto::TMulSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getDst().getType(),
      getScalar().getType(), /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmuls element type to be i32/i16/f16/f32",
      "expects A5 tmuls element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

mlir::LogicalResult mlir::pto::TShlSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem)
    return emitOpError() << "failed to get element type for src/dst";
  if (srcElem != dstElem)
    return emitOpError() << "expects src and dst to have the same element type";
  if (!mlir::isa<IntegerType>(srcElem))
    return emitOpError() << "expects integral element types";
  if (auto scalarValue = getConstantIntegerValue(getScalar()); scalarValue && *scalarValue < 0)
    return emitOpError("expects tshls scalar to be non-negative");
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TShrSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem) {
      emitOpError("failed to get element type for src/dst");
      return failure();
    }
    if (srcElem != dstElem) {
      emitOpError("expects src and dst to have the same element type");
      return failure();
    }
    return srcElem;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 16 && it.getWidth() != 32))
      return emitOpError(
          "expects A2/A3 tshrs src and dst element type to be i16/i32");
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
          "expects A5 tshrs src and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TNegOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(16) || elemTy.isInteger(32) || elemTy.isF16() ||
          elemTy.isF32()))
      return emitOpError()
             << "expects A2/A3 tneg element type to be i16/i32/f16/f32";
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")))
      return failure();

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return emitOpError() << "expects src and dst to have rank-2 valid_shape";
    if (srcValid[1] != ShapedType::kDynamic &&
        dstValid[1] != ShapedType::kDynamic &&
        srcValid[1] != dstValid[1])
      return emitOpError()
             << "expects src and dst to have the same valid_shape[1]";

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32() || elemTy.isBF16()))
      return emitOpError()
             << "expects A5 tneg element type to be i8/i16/i32/f16/f32/bf16";
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TNotOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    auto elemTy = getElemTy(srcTy);
    if (elemTy != getElemTy(dstTy))
      return emitOpError() << "expects src and dst to have the same element type";
    if (!elemTy.isInteger(16))
      return emitOpError() << "expects A2/A3 tnot element type to be i16";
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    auto elemTy = getElemTy(srcTy);
    if (elemTy != getElemTy(dstTy))
      return emitOpError() << "expects src and dst to have the same element type";
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32)))
      return emitOpError() << "expects A5 tnot element type to be i8/i16/i32";
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TOrOp::verify() {
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
          "expects A2/A3 tor src0, src1, and dst element type to be i8/i16");
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
          "expects A5 tor src0, src1, and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TOrSOp::verify() {
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
          "expects A2/A3 tors src and dst element type to be i8/i16");
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
          "expects A5 tors src and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyPTOShapedBinarySameElemAndShape(Operation *op,
                                                              Type src0Ty,
                                                              Type src1Ty,
                                                              Type dstTy) {
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy))
    return op->emitOpError(
               "expects src0/src1/dst to be memref/tensor/tile_buf/tile_view types"),
           failure();
  Type e0 = getElemTy(src0Ty), e1 = getElemTy(src1Ty), ed = getElemTy(dstTy);
  if (!e0 || !e1 || !ed)
    return op->emitOpError("failed to get element type for operands"), failure();
  if (e0 != e1 || e0 != ed)
    return op->emitOpError("expects src0/src1/dst to have the same element type"),
           failure();
  auto s0 = getShapeVec(src0Ty), s1 = getShapeVec(src1Ty), sd = getShapeVec(dstTy);
  if (s0 != s1 || s0 != sd)
    return op->emitOpError("expects src0/src1/dst to have the same shape"),
           failure();
  return e0;
}

mlir::LogicalResult mlir::pto::TPartAddOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
        !isPTOShapedLike(dstTy))
      return emitOpError() << "expects PTO shaped-like src0/src1/dst";
    if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
        getElemTy(src0Ty) != getElemTy(dstTy))
      return emitOpError() << "expects src0/src1/dst to have the same element type";
    auto s0 = getShapeVec(src0Ty);
    auto s1 = getShapeVec(src1Ty);
    auto d = getShapeVec(dstTy);
    if (s0.size() != 2 || s1.size() != 2 || d.size() != 2)
      return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped)";
    if (failed(verifyPartialValidPattern(*this, src0Ty, src1Ty, dstTy)))
      return failure();
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tpartadd element type to be i32/i16/f16/f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
        !isPTOShapedLike(dstTy))
      return emitOpError() << "expects PTO shaped-like src0/src1/dst";
    if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
        getElemTy(src0Ty) != getElemTy(dstTy))
      return emitOpError() << "expects src0/src1/dst to have the same element type";
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32()))
      return emitOpError("expects A5 tpartadd element type to be i32/i16/i8/f16/bf16/f32");
    auto s0 = getShapeVec(src0Ty);
    auto s1 = getShapeVec(src1Ty);
    auto d = getShapeVec(dstTy);
    if (s0.size() != 2 || s1.size() != 2 || d.size() != 2)
      return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped)";
    if (failed(verifyPartialValidPatternLoose(*this, src0Ty, src1Ty, dstTy)))
      return failure();
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TPartMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr))
      return failure();
    if (failed(verifyPartialValidPattern(*this, t0, t1, td)))
      return failure();
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isF16() || e0.isF32()))
      return emitOpError("expects A2/A3 tpartmax element type to be i32/i16/f16/f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr))
      return failure();
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isInteger(8) ||
          e0.isF16() || e0.isBF16() || e0.isF32()))
      return emitOpError("expects A5 tpartmax element type to be i32/i16/i8/f16/bf16/f32");
    if (failed(verifyPartialValidPatternLoose(*this, t0, t1, td)))
      return failure();
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TPartMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr))
      return failure();
    if (failed(verifyPartialValidPattern(*this, t0, t1, td)))
      return failure();
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isF16() || e0.isF32()))
      return emitOpError("expects A2/A3 tpartmin element type to be i32/i16/f16/f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr))
      return failure();
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isInteger(8) ||
          e0.isF16() || e0.isBF16() || e0.isF32()))
      return emitOpError("expects A5 tpartmin element type to be i32/i16/i8/f16/bf16/f32");
    if (failed(verifyPartialValidPatternLoose(*this, t0, t1, td)))
      return failure();
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPartArgOpCommon(Operation *op, Type src0Ty,
                                            Type src1Ty, Type src0IdxTy,
                                            Type src1IdxTy, Type dstTy,
                                            Type dstIdxTy, StringRef opName) {
  FailureOr<Type> dataElemOr =
      verifyPTOShapedBinarySameElemAndShape(op, src0Ty, src1Ty, dstTy);
  if (failed(dataElemOr))
    return failure();
  if (failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy)))
    return failure();

  if (!isPTOShapedLike(src0IdxTy) || !isPTOShapedLike(src1IdxTy) ||
      !isPTOShapedLike(dstIdxTy))
    return op->emitOpError("expects PTO shaped-like src0Idx/src1Idx/dstIdx");
  Type idxElem = getElemTy(src0IdxTy);
  if (!idxElem || idxElem != getElemTy(src1IdxTy) ||
      idxElem != getElemTy(dstIdxTy))
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx to have the same element type");
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != 32)
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx element type to be i32 or ui32");

  auto dataShape = getShapeVec(src0Ty);
  if (dataShape != getShapeVec(src0IdxTy) ||
      dataShape != getShapeVec(src1IdxTy) ||
      dataShape != getShapeVec(dstIdxTy))
    return op->emitOpError(
        "expects data and index operands to have the same shape");
  if (getValidShapeVec(src0Ty) != getValidShapeVec(src0IdxTy) ||
      getValidShapeVec(src1Ty) != getValidShapeVec(src1IdxTy) ||
      getValidShapeVec(dstTy) != getValidShapeVec(dstIdxTy))
    return op->emitOpError(
        "expects each data operand and its index operand to have the same valid_shape");

  Type elem = *dataElemOr;
  PTOArch arch = getTargetArch(op);
  if (arch == PTOArch::A5) {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32()))
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i32/i16/i8/f16/bf16/f32";
  } else {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32()))
      return op->emitOpError() << "expects A2/A3 " << opName
                               << " element type to be i32/i16/f16/f32";
  }
  return success();
}

mlir::LogicalResult mlir::pto::TPartArgMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTPartArgOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getSrc0Idx().getType(), getSrc1Idx().getType(), getDst().getType(),
        getDstIdx().getType(), "tpartargmax");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TPartArgMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTPartArgOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getSrc0Idx().getType(), getSrc1Idx().getType(), getDst().getType(),
        getDstIdx().getType(), "tpartargmin");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TPartMulOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
        !isPTOShapedLike(dstTy))
      return emitOpError() << "expects PTO shaped-like src0/src1/dst";
    if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
        getElemTy(src0Ty) != getElemTy(dstTy))
      return emitOpError()
             << "expects src0/src1/dst to have the same element type";
    auto s0 = getShapeVec(src0Ty);
    auto s1 = getShapeVec(src1Ty);
    auto d = getShapeVec(dstTy);
    if (s0.size() != 2 || s1.size() != 2 || d.size() != 2)
      return emitOpError()
             << "expects src0/src1/dst to be rank-2 (tile-shaped)";
    if (failed(verifyPartialValidPattern(*this, src0Ty, src1Ty, dstTy)))
      return failure();
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32()))
      return emitOpError(
          "expects A2/A3 tpartmul element type to be i32/i16/f16/f32");
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
        !isPTOShapedLike(dstTy))
      return emitOpError() << "expects PTO shaped-like src0/src1/dst";
    if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
        getElemTy(src0Ty) != getElemTy(dstTy))
      return emitOpError()
             << "expects src0/src1/dst to have the same element type";
    Type elem = getElemTy(src0Ty);
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32()))
      return emitOpError(
          "expects A5 tpartmul element type to be i32/i16/i8/f16/bf16/f32");
    auto s0 = getShapeVec(src0Ty);
    auto s1 = getShapeVec(src1Ty);
    auto d = getShapeVec(dstTy);
    if (s0.size() != 2 || s1.size() != 2 || d.size() != 2)
      return emitOpError()
             << "expects src0/src1/dst to be rank-2 (tile-shaped)";
    if (failed(verifyPartialValidPatternLoose(*this, src0Ty, src1Ty, dstTy)))
      return failure();
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TPReluOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto verifyCommon = [&]() -> FailureOr<std::tuple<Type, Type, Type, Type>> {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type tt = getTmp().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, tt, "tmp")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();

    Type e0 = getElemTy(t0), e1 = getElemTy(t1), et = getElemTy(tt), ed = getElemTy(td);
    if (!e0 || !e1 || !et || !ed) {
      emitOpError("failed to get element type for operands");
      return failure();
    }
    if (e0 != e1 || e0 != ed) {
      emitOpError("expects dst/src0/src1 to have the same element type");
      return failure();
    }
    if (!(e0.isF16() || e0.isF32())) {
      emitOpError("expects dst/src0/src1 element type to be f16 or f32");
      return failure();
    }
    if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) || !isRowMajorTileBuf(td)) {
      emitOpError("expects src0, src1, and dst to use row-major layout");
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, t0, td, "src0", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, t1, td, "src1", "dst")))
      return failure();

    auto s0 = getShapeVec(t0), s1 = getShapeVec(t1), st = getShapeVec(tt), sd = getShapeVec(td);
    if (s0 != s1 || s0 != st || s0 != sd) {
      emitOpError("expects src0/src1/tmp/dst to have the same shape");
      return failure();
    }
    return std::make_tuple(t0, t1, tt, td);
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto tysOr = verifyCommon();
    if (failed(tysOr))
      return failure();
    auto [t0, t1, tt, td] = *tysOr;
    Type tmpElem = getElemTy(tt);
    auto tmpIntTy = mlir::dyn_cast<IntegerType>(tmpElem);
    if (!tmpIntTy || tmpIntTy.getWidth() != 8)
      return emitOpError("expects A2/A3 tmp element type to be u8");
    if (!isRowMajorTileBuf(tt))
      return emitOpError("expects tmp to use row-major layout");
    if (auto arch = getVerifierArchName(getOperation());
        arch && arch->equals_insensitive("a3")) {
      if (getSrc0() == getSrc1() || getSrc0() == getTmp() || getSrc0() == getDst() ||
          getSrc1() == getTmp() || getSrc1() == getDst() || getTmp() == getDst())
        return emitOpError(
            "expects A3 src0, src1, tmp, and dst to use different storage");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto tysOr = verifyCommon();
    if (failed(tysOr))
      return failure();
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TQuantOp::verify() {
  // Structural checks: always run regardless of operand representation
  // (applies both before and after PTOViewToMemref lowering).
  auto verifyStructural = [&]() -> LogicalResult {
    // dst elem type and offset presence must be consistent with quant_type.
    Type dstTy = getDst().getType();
    Type dstElemTy = getElemTy(dstTy);
    auto dstIntTy = dyn_cast<IntegerType>(dstElemTy);
    if (getQuantType() == mlir::pto::QuantType::INT8_SYM) {
      if (!dstIntTy || dstIntTy.getWidth() != 8)
        return emitOpError()
               << "expects dst element type i8/ui8 for INT8_SYM quantization";
      if (getOffset())
        return emitOpError()
               << "INT8_SYM quantization must not have an offset operand";
    } else {
      // INT8_ASYM
      if (!dstIntTy || dstIntTy.getWidth() != 8)
        return emitOpError()
               << "expects dst element type i8/ui8 for INT8_ASYM quantization";
      if (!getOffset())
        return emitOpError()
               << "INT8_ASYM quantization requires an offset operand";
    }
    return success();
  };

  if (failed(verifyStructural()))
    return failure();

  // Layout/tile-buffer checks: only meaningful for pre-lowering tile types.
  // Skip when operands are already plain MemRefs (post PTOViewToMemref).
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  auto verifyCommon = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type fpTy  = getFp().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    // src must be f32 (ISA static_assert)
    if (!getElemTy(srcTy).isF32())
      return emitOpError() << "expects src to have element type f32";
    if (getOffset()) {
      Type offsetTy = getOffset().getType();
      if (failed(verifyTileBufCommon(*this, offsetTy, "offset")))
        return failure();
      if (!getElemTy(offsetTy).isF32())
        return emitOpError() << "expects offset to have element type f32";
    }
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyCommon()))
      return failure();
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy))
      return emitOpError() << "expects A2/A3 src and dst to use row-major layout";
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    return verifyCommon();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDequantOp::verify() {
  // Structural checks: src must be i8 or i16, dst/scale/offset must be f32.
  auto verifyStructural = [&]() -> LogicalResult {
    Type srcElemTy = getElemTy(getSrc().getType());
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!srcIntTy || !(srcIntTy.getWidth() == 8 || srcIntTy.getWidth() == 16))
      return emitOpError()
             << "expects src element type i8 or i16";
    if (!getElemTy(getDst().getType()).isF32())
      return emitOpError() << "expects dst element type f32";
    if (!getElemTy(getScale().getType()).isF32())
      return emitOpError() << "expects scale element type f32";
    if (!getElemTy(getOffset().getType()).isF32())
      return emitOpError() << "expects offset element type f32";
    return success();
  };

  if (failed(verifyStructural()))
    return failure();

  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  auto verifyCommon = [&]() -> LogicalResult {
    if (failed(verifyTileBufCommon(*this, getSrc().getType(), "src")) ||
        failed(verifyTileBufCommon(*this, getScale().getType(), "scale")) ||
        failed(verifyTileBufCommon(*this, getOffset().getType(), "offset")) ||
        failed(verifyTileBufCommon(*this, getDst().getType(), "dst")))
      return failure();
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyCommon()))
      return failure();
    if (!isRowMajorTileBuf(getSrc().getType()) ||
        !isRowMajorTileBuf(getDst().getType()))
      return emitOpError()
             << "expects A2/A3 src and dst to use row-major layout";
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult { return verifyCommon(); };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRecipOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst")))
    return failure();
  Type elemTy = getElemTy(ts);
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";
  if (auto arch = getVerifierArchName(getOperation());
      arch && arch->equals_insensitive("a3") && getSrc() == getDst())
    return emitOpError("expects A3 trecip src and dst to use different storage");
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TReluOp::verify() {
  auto verifyByArch = [&](StringRef errorMessage) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
      return failure();
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(32) || elemTy.isF16() || elemTy.isF32()))
      return emitOpError() << errorMessage;
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch("expects A2/A3 trelu element type to be i32/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch("expects A5 trelu element type to be i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRemOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type tmpTy = getTmp().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(*this, tmpTy, "tmp")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(*this, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameElemType(*this, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
    return failure();
  if (getElemTy(tmpTy) != getElemTy(dstTy))
    return emitOpError("expects tmp and dst to have the same element type");
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(tmpTy) || !isRowMajorTileBuf(dstTy))
    return emitOpError("expects src0, src1, tmp, and dst to use row-major layout");
  auto dstValid = getValidShapeVec(dstTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (dstValid.size() != 2 || tmpValid.size() != 2)
    return emitOpError("expects tmp and dst to be rank-2 tiles");
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 0)
    return emitOpError("expects tmp to have non-negative valid rows");
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
      tmpValid[1] < dstValid[1])
    return emitOpError("expects tmp valid columns to cover dst valid columns");

  Type elem = getElemTy(src0Ty);
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isF32()))
      return emitOpError("expects A2/A3 trem element type to be i32/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 trem element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TFModOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tfmod element type to be i32/i16/f16/f32",
      "expects A5 tfmod element type to be i32/i16/f16/f32");
}

mlir::LogicalResult mlir::pto::TRemSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type tt = getTmp().getType();
  Type td = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, ts, "src")) ||
      failed(verifyTileBufCommon(*this, tt, "tmp")) ||
      failed(verifyTileBufCommon(*this, td, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(*this, ts, td, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst")))
    return failure();
  if (getElemTy(tt) != getElemTy(td))
    return emitOpError("expects tmp and dst to have the same element type");
  if (!isRowMajorTileBuf(ts) || !isRowMajorTileBuf(tt) || !isRowMajorTileBuf(td))
    return emitOpError("expects src, tmp, and dst to use row-major layout");
  Type elem = getElemTy(ts);
  if (scalarTy != elem)
    return emitOpError("expects scalar type to match the tile element type");
  auto dstValid = getValidShapeVec(td);
  auto tmpValid = getValidShapeVec(tt);
  if (dstValid.size() != 2 || tmpValid.size() != 2)
    return emitOpError("expects tmp and dst to be rank-2 tiles");
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 0)
    return emitOpError("expects tmp to have non-negative valid rows");
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
      tmpValid[1] < dstValid[1])
    return emitOpError("expects tmp valid columns to cover dst valid columns");
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isF32()))
      return emitOpError("expects A2/A3 trems element type to be i32/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 trems element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TFModSOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();

  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst")))
    return failure();
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy))
    return emitOpError("expects src and dst to use row-major layout");

  Type elem = getElemTy(srcTy);
  if (scalarTy != elem)
    return emitOpError("expects scalar type to match the tile element type");

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A2/A3 tfmods element type to be i32/i16/f16/f32");
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32()))
      return emitOpError("expects A5 tfmods element type to be i32/i16/f16/f32");
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static std::optional<int64_t> getStaticNumElements(ArrayRef<int64_t> shape) {
  int64_t numel = 1;
  for (int64_t d : shape) {
    if (d == ShapedType::kDynamic)
      return std::nullopt;
    if (d < 0)
      return std::nullopt;
    numel *= d;
  }
  return numel;
}

static std::optional<int64_t> getElemBytes(Type elemTy) {
  if (!elemTy)
    return std::nullopt;
  if (auto ft = dyn_cast<FloatType>(elemTy)) {
    if (ft.isF16() || ft.isBF16())
      return 2;
    if (ft.isF32())
      return 4;
    if (ft.isF64())
      return 8;
    return std::nullopt;
  }
  if (auto it = dyn_cast<IntegerType>(elemTy)) {
    int64_t bits = it.getWidth();
    if (bits <= 0)
      return std::nullopt;
    return std::max<int64_t>(1, bits / 8);
  }
  return std::nullopt;
}

[[maybe_unused]] static bool isTileBufOrMemref(Type ty) {
  return mlir::isa<MemRefType, pto::TileBufType>(ty);
}

static constexpr llvm::StringLiteral kLoweredSetValidShapeAttrName =
    "__pto.lowered_set_validshape";

static bool isLocallyBoundTileSource(Value value) {
  if (!value || isa<BlockArgument>(value))
    return false;

  if (isa<AllocTileOp, DeclareTileOp, BindTileOp, PointerCastOp,
          MaterializeTileOp>(
          value.getDefiningOp()))
    return true;

  if (auto bitcast = value.getDefiningOp<BitcastOp>())
    return isLocallyBoundTileSource(bitcast.getSrc());
  if (auto reshape = value.getDefiningOp<TReshapeOp>())
    return isLocallyBoundTileSource(reshape.getSrc());

  return false;
}

static std::optional<int64_t> getConstIndexLike(Value v) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>())
    return cOp.value();
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>())
    return cInt.value();
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue()))
      return ia.getInt();
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndexLike(castOp.getIn());
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
    return getConstIndexLike(extOp.getIn());
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
    return getConstIndexLike(extOp.getIn());
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>())
    return getConstIndexLike(truncOp.getIn());
  return std::nullopt;
}

mlir::LogicalResult mlir::pto::SetValidShapeOp::verify() {
  SmallVector<int64_t> shape;
  if (auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType())) {
    if (srcTy.getRank() != 2)
      return emitOpError("expects rank-2 tile_buf source");

    ArrayRef<int64_t> validShape = srcTy.getValidShape();
    if (validShape.size() != 2)
      return emitOpError("expects source validShape to be rank-2");
    if (!srcTy.hasDynamicValid())
      return emitOpError("expects source tile_buf to have dynamic validShape (?, ?)");

    shape.assign(srcTy.getShape().begin(), srcTy.getShape().end());

    if (!isLocallyBoundTileSource(getSource()))
      return emitOpError(
          "requires a locally bound tile source; function arguments/results "
          "are unsupported");
  } else if (auto srcTy = llvm::dyn_cast<MemRefType>(getSource().getType())) {
    if (!(*this)->hasAttr(kLoweredSetValidShapeAttrName))
      return emitOpError(
          "expects tile_buf source; memref source is only valid for the internal lowered form");
    if (srcTy.getRank() != 2)
      return emitOpError("expects rank-2 memref source after tile lowering");
    shape.assign(srcTy.getShape().begin(), srcTy.getShape().end());
  } else {
    return emitOpError("expects tile_buf source (or lowered memref source)");
  }

  auto checkDim = [&](Value operand, unsigned dimIdx,
                      StringRef dimName) -> LogicalResult {
    int64_t maxStatic = shape[dimIdx];

    auto constVal = getConstIndexLike(operand);
    if (!constVal)
      return success();

    if (*constVal < 0)
      return emitOpError() << "expects " << dimName << " operand to be non-negative";
    if (maxStatic != ShapedType::kDynamic && *constVal > maxStatic)
      return emitOpError() << "expects " << dimName << " operand <= shape dim ("
                           << maxStatic << ")";
    return success();
  };

  if (failed(checkDim(getValidRow(), /*dimIdx=*/0, "row")))
    return failure();
  if (failed(checkDim(getValidCol(), /*dimIdx=*/1, "col")))
    return failure();

  return success();
}

mlir::LogicalResult mlir::pto::GetValidShapeOp::verify() {
  if (auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType())) {
    if (srcTy.getRank() != 2)
      return emitOpError("expects rank-2 tile_buf source");
    if (srcTy.getValidShape().size() != 2)
      return emitOpError("expects source validShape to be rank-2");
    return success();
  }
  if (auto srcTy = llvm::dyn_cast<MemRefType>(getSource().getType())) {
    if (srcTy.getRank() != 2)
      return emitOpError("expects rank-2 memref source after tile lowering");
    return success();
  }
  return emitOpError("expects tile_buf source (or lowered memref source)");
}


mlir::LogicalResult mlir::pto::TReshapeOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type tr = getResult().getType();
  auto srcTb = dyn_cast<pto::TileBufType>(ts);
  auto dstTb = dyn_cast<pto::TileBufType>(tr);
  if (!srcTb || !dstTb)
    return emitOpError("expects src/result to be !pto.tile_buf types");

  if (failed(verifyTileBufCommon(*this, ts, "src")) ||
      failed(verifyTileBufCommon(*this, tr, "dst")))
    return failure();

  if (srcTb.getMemorySpace() != dstTb.getMemorySpace())
    return emitOpError("expects src and dst to use the same loc");

  Type srcElem = srcTb.getElementType();
  Type dstElem = dstTb.getElementType();
  auto srcElemBytes = getElemBytes(srcElem);
  auto dstElemBytes = getElemBytes(dstElem);
  if (!srcElem || !dstElem || !srcElemBytes.has_value() || !dstElemBytes.has_value())
    return emitOpError("failed to get element byte width for src/dst");

  auto srcNumel = getStaticNumElements(getShapeVec(ts));
  auto dstNumel = getStaticNumElements(getShapeVec(tr));
  if (!srcNumel.has_value() || !dstNumel.has_value())
    return emitOpError("expects static shapes for treshape");

  if (srcElemBytes.value() * srcNumel.value() !=
      dstElemBytes.value() * dstNumel.value())
    return emitOpError("expects src and dst to have the same total byte size");

  bool srcBoxed =
      srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  bool dstBoxed =
      dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  if (srcBoxed != dstBoxed)
    return emitOpError("cannot reshape between boxed and non-boxed tile layouts");

  return success();
}

mlir::LogicalResult mlir::pto::BitcastOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto srcTy = llvm::dyn_cast<TileBufType>(getSrc().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects tile_buf src and tile_buf result");

  if (srcTy.getMemorySpace() != dstTy.getMemorySpace())
    return emitOpError("expects src/result to have the same memorySpace");

  if (srcTy.getElementType() == dstTy.getElementType())
    return emitOpError(
        "expects src/result to have different element types; use "
        "pto.treshape for shape/config changes");

  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("expects src/result to have the same shape; use pto.treshape for shape changes");

  if (srcTy.getValidShape() != dstTy.getValidShape())
    return emitOpError("expects src/result to have the same validShape");

  auto srcCfg = srcTy.getConfigAttr();
  auto dstCfg = dstTy.getConfigAttr();
  if (srcCfg != dstCfg)
    return emitOpError("expects src/result to have the same tile config");

  auto numel = getStaticNumElements(srcTy.getShape());
  if (!numel.has_value())
    return emitOpError("expects static shapes for bitcast");

  auto srcBytes = getElemBytes(srcTy.getElementType());
  auto dstBytes = getElemBytes(dstTy.getElementType());
  if (!srcBytes.has_value() || !dstBytes.has_value())
    return emitOpError("unsupported element type for bitcast");

  int64_t srcTotalBytes = numel.value() * srcBytes.value();
  int64_t dstTotalBytes = numel.value() * dstBytes.value();
  if (dstTotalBytes > srcTotalBytes)
    return emitOpError("bitcast result requires more bytes than source storage");

  return success();
}


mlir::LogicalResult mlir::pto::TRowExpandOp::verify() {
  auto verifyCommon = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst")))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::VEC)
      return emitOpError("expects src to be in the vec address space");
    if (auto srcTb = dyn_cast<pto::TileBufType>(srcTy)) {
      if (srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox))
        return emitOpError("expects src to use the none_box slayout");
    }
    if (getElemTy(srcTy) != getElemTy(dstTy))
      return emitOpError("expects src and dst to have the same element type");
    if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                                /*allowInt8=*/true))
      return emitOpError("expects trowexpand element type to be supported");
    auto srcValid = getValidShapeVec(getSrc());
    auto dstValid = getValidShapeVec(getDst());
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return emitOpError("expects src and dst to have rank-2 valid_shape");
    // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. The op
    // writes no elements; accept and skip the non-empty constraints. One-sided
    // empties still fall through. See pto-isa#143 for hardware Rv=0 no-op.
    if (dstValid[0] == 0 && dstValid[1] == 0)
      return success();
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0])
      return emitOpError("expects src and dst to have the same valid_shape[0]");
    if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0)
      return emitOpError("expects src valid_shape[0] to be non-zero");
    if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0)
      return emitOpError("expects src valid_shape[1] to be non-zero");
    if (dstValid[0] != ShapedType::kDynamic && dstValid[0] == 0)
      return emitOpError("expects dst valid_shape[0] to be non-zero");
    if (dstValid[1] != ShapedType::kDynamic && dstValid[1] == 0)
      return emitOpError("expects dst valid_shape[1] to be non-zero");
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyCommon();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyCommon();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


ParseResult mlir::pto::TSort32Op::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, idx, tmp, dst;
  Type srcTy, dstTy, idxTy, tmpTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(idx))
      return failure();
    if (succeeded(parser.parseOptionalComma())) {
      if (parser.parseOperand(tmp))
        return failure();
      hasTmp = true;
    }
  } else {
    return failure();
  }
  if (parser.parseColonType(srcTy) || parser.parseComma() || parser.parseType(idxTy))
    return failure();
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy))
      return failure();
  }
  if (parser.parseRParen())
    return failure();

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(idx, idxTy, result.operands))
    return failure();
  if (hasTmp) {
    if (parser.resolveOperand(tmp, tmpTy, result.operands))
      return failure();
  }
  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, 1, hasTmp ? 1 : 0, 1}));
  return success();
}

void mlir::pto::TSort32Op::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getIdx();
  if (getTmp()) {
    p << ", " << getTmp();
    p << " : " << getSrc().getType() << ", " << getIdx().getType()
      << ", " << getTmp().getType() << ")";
  } else {
    p << " : " << getSrc().getType() << ", " << getIdx().getType() << ")";
  }
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TRsqrtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, tmp, dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp))
      return failure();
    hasTmp = true;
  }
  if (parser.parseColonType(srcTy))
    return failure();
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy))
      return failure();
  }
  if (parser.parseRParen())
    return failure();

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands))
    return failure();
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands))
    return failure();

  return success();
}

void mlir::pto::TRsqrtOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (getTmp())
    p << ", " << getTmp();
  p << " : " << getSrc().getType();
  if (getTmp())
    p << ", " << getTmp().getType();
  p << ")";
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs());
}

static ParseResult parseTRowExpandBinaryLikeOp(OpAsmParser &parser,
                                               OperationState &result) {
  OpAsmParser::UnresolvedOperand src0, src1, tmp, dst;
  Type src0Ty, src1Ty, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src0) || parser.parseComma() || parser.parseOperand(src1))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp))
      return failure();
    hasTmp = true;
  }
  if (parser.parseColon())
    return failure();
  if (parser.parseType(src0Ty) || parser.parseComma() || parser.parseType(src1Ty))
    return failure();
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy))
      return failure();
  }
  if (parser.parseRParen())
    return failure();
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.resolveOperand(src0, src0Ty, result.operands) ||
      parser.resolveOperand(src1, src1Ty, result.operands))
    return failure();
  if (hasTmp) {
    if (parser.resolveOperand(tmp, tmpTy, result.operands))
      return failure();
  }
  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, 1, hasTmp ? 1 : 0, 1}));
  return success();
}

static void printTRowExpandBinaryLikeOp(OpAsmPrinter &p, Operation *op, Value src0,
                                        Value src1, Value tmp, Value dst) {
  p << " ins(" << src0 << ", " << src1;
  if (tmp) {
    p << ", " << tmp;
    p << " : " << src0.getType() << ", " << src1.getType() << ", "
      << tmp.getType() << ")";
  } else {
    p << " : " << src0.getType() << ", " << src1.getType() << ")";
  }
  p << " outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TRowExpandDivOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandDivOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMulOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMulOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandSubOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandSubOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandExpdifOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandExpdifOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMaxOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMaxOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMinOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMinOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

static FailureOr<Type> verifyTRowExpandBinaryCore(Operation *op, Type src0Ty,
                                                  Type src1Ty, Type dstTy,
                                                  Type tmpTy, bool hasTmp) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  if (hasTmp && failed(verifyTileBufCommon(op, tmpTy, "tmp")))
    return failure();
  if (failed(verifyTileBufSameElemType(op, src0Ty, dstTy, "src0", "dst")))
    return failure();
  if (getElemTy(src0Ty) != getElemTy(src1Ty)) {
    op->emitOpError("expects src0 and src1 to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects dst to use row-major layout");
    return failure();
  }
  return getElemTy(src0Ty);
}

mlir::LogicalResult mlir::pto::TRowExpandDivOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr))
      return failure();
    Type elem = *elemOr;
    bool supported =
        elem.isF16() || elem.isF32() ||
        (targetArch == PTOArch::A5 &&
         (elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32)));
    if (!supported) {
      if (targetArch == PTOArch::A5)
        return emitOpError(
            "expects A5 trowexpanddiv element type to be i8/i16/i32/f16/f32");
      return emitOpError("expects element type to be f16 or f32");
    }
    if (getPrecisionType() == pto::DivPrecision::HighPrecision && !getTmp())
      return emitOpError("expects tmp when precisionType is high_precision");
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandMulOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr))
      return failure();
    Type elem = *elemOr;
    bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                     elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8));
    if (!supported) {
      if (targetArch == PTOArch::A5)
        return emitOpError(
            "expects A5 trowexpandmul element type to be i8/i16/i32/f16/f32");
      return emitOpError(
          "expects A2/A3 trowexpandmul element type to be i16/i32/f16/f32");
    }
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandSubOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr))
      return failure();
    Type elem = *elemOr;
    bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                     elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8));
    if (!supported) {
      if (targetArch == PTOArch::A5)
        return emitOpError(
            "expects A5 trowexpandsub element type to be i8/i16/i32/f16/f32");
      return emitOpError(
          "expects A2/A3 trowexpandsub element type to be i16/i32/f16/f32");
    }
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandAddOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
        failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    if (failed(verifyTileBufSameElemType(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst")))
      return failure();
    if (getElemTy(src0Ty) != getElemTy(src1Ty))
      return emitOpError("expects src0 and src1 to have the same element type");
    if (!isRowMajorTileBuf(src0Ty))
      return emitOpError("expects src0 to use row-major layout");
    if (!isRowMajorTileBuf(dstTy))
      return emitOpError("expects dst to use row-major layout");
    Type elem = getElemTy(src0Ty);
    bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                     elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8));
    if (!supported) {
      if (targetArch == PTOArch::A5)
        return emitOpError(
            "expects A5 trowexpandadd element type to be i8/i16/i32/f16/f32");
      return emitOpError(
          "expects A2/A3 trowexpandadd element type to be i16/i32/f16/f32");
    }
    auto src1Valid = getValidShapeVec(src1Ty);
    auto dstValid = getValidShapeVec(dstTy);
    if (src1Valid.size() != 2 || dstValid.size() != 2)
      return emitOpError("expects src1 and dst to have rank-2 valid_shape");
    if (src1Valid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        src1Valid[0] != dstValid[0])
      return emitOpError("expects src1 valid_shape[0] to equal dst valid_shape[0]");
    bool src1IsRowMajor = isRowMajorTileBuf(src1Ty);
    int64_t expectedCol = elem.isInteger(8)
                              ? 32
                              : ((elem.isF16() || elem.isInteger(16)) ? 16 : 8);
    int64_t src1Col = src1Valid[1];
    if (src1IsRowMajor) {
      if (src1Col != ShapedType::kDynamic && src1Col != expectedCol)
        return emitOpError("expects row-major src1 valid_shape[1] to be 32/sizeof(dtype)");
    } else {
      if (src1Col != ShapedType::kDynamic && src1Col != 1)
        return emitOpError("expects non-row-major src1 valid_shape[1] to be 1");
    }
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTRowExpandReduceLikeOp(Operation *op, Type src0Ty,
                                                  Type src1Ty, Type dstTy,
                                                  Type tmpTy, bool hasTmp,
                                                  PTOArch targetArch,
                                                  StringRef opName,
                                                  bool allowIntegerTypes) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")))
    return failure();
  if (hasTmp) {
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp")))
      return failure();
    if (getElemTy(tmpTy) != getElemTy(dstTy))
      return op->emitOpError() << "expects tmp and dst to have the same element type";
  }

  Type elem = getElemTy(dstTy);
  if (!elem || getElemTy(src0Ty) != elem || getElemTy(src1Ty) != elem)
    return op->emitOpError("expects src0, src1, and dst to have the same element type");
  bool supported = elem.isF16() || elem.isF32() ||
                   (allowIntegerTypes &&
                    (elem.isInteger(16) || elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8))));
  if (!supported) {
    if (!allowIntegerTypes)
      return op->emitOpError() << "expects " << opName
                               << " element type to be f16 or f32";
    if (targetArch == PTOArch::A5)
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i8/i16/i32/f16/f32";
    return op->emitOpError() << "expects A2/A3 " << opName
                             << " element type to be i16/i32/f16/f32";
  }

  if (!isRowMajorTileBuf(dstTy))
    return op->emitOpError("expects dst to use row-major layout");

  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2)
    return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");

  // Operand-form invariant, enforced regardless of the valid region: A5 has no
  // tmp form for these ops. Must run before the empty-marker early-accept below,
  // otherwise a 0x0 dst would let an A5 tmp form slip through and lower to the
  // A2/A3 4-operand TROWEXPAND* call.
  if (hasTmp && targetArch == PTOArch::A5)
    return op->emitOpError("expects A5 form to omit tmp");

  // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. Element
  // type/layout were already checked above; the op writes no elements, so accept
  // and skip the non-empty broadcast/width constraints. One-sided empties still
  // fall through. See pto-isa#143 for hardware Rv=0 no-op.
  if (dstValid[0] == 0 && dstValid[1] == 0)
    return success();

  if (dstValid[0] != ShapedType::kDynamic && dstValid[0] == 0)
    return op->emitOpError("expects dst valid_shape[0] to be non-zero");
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] == 0)
    return op->emitOpError("expects dst valid_shape[1] to be non-zero");

  auto validShapeMatches = [](ArrayRef<int64_t> lhs,
                              ArrayRef<int64_t> rhs) -> bool {
    if (lhs.size() != rhs.size())
      return false;
    for (auto [l, r] : llvm::zip(lhs, rhs)) {
      if (l != ShapedType::kDynamic && r != ShapedType::kDynamic && l != r)
        return false;
    }
    return true;
  };

  const bool src0MatchesDst = validShapeMatches(src0Valid, dstValid);
  const bool src1MatchesDst = validShapeMatches(src1Valid, dstValid);

  auto checkBroadcastOperand = [&](Type operandTy, ArrayRef<int64_t> operandValid,
                                   StringRef operandName,
                                   bool requireNonRowMajor) -> LogicalResult {
    if (operandValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        operandValid[0] != dstValid[0]) {
      return op->emitOpError() << "expects " << operandName
                               << " valid_shape[0] to equal dst valid_shape[0]";
    }
    int64_t expectedCol = elem.isInteger(8) ? 32 : ((elem.isF16() || elem.isInteger(16)) ? 16 : 8);
    int64_t operandCol = operandValid[1];
    bool operandIsRowMajor = isRowMajorTileBuf(operandTy);
    if (requireNonRowMajor && operandIsRowMajor) {
      return op->emitOpError() << "expects " << operandName
                               << " to use a non-row-major layout when tmp is present";
    }
    if (operandIsRowMajor) {
      if (operandCol != ShapedType::kDynamic && operandCol != expectedCol) {
        return op->emitOpError()
               << "expects row-major " << operandName
               << " valid_shape[1] to be 32/sizeof(dtype)";
      }
      return success();
    }
    if (operandCol != ShapedType::kDynamic && operandCol != 1) {
      return op->emitOpError() << "expects non-row-major " << operandName
                               << " valid_shape[1] to be 1";
    }
    return success();
  };

  auto checkFullAndBroadcast = [&](Type fullTy, ArrayRef<int64_t> fullValid,
                                   StringRef fullName, Type broadcastTy,
                                   ArrayRef<int64_t> broadcastValid,
                                   StringRef broadcastName) -> LogicalResult {
    if (!isRowMajorTileBuf(fullTy))
      return op->emitOpError() << "expects " << fullName
                               << " to use row-major layout when it matches dst";
    if (fullValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        fullValid[0] != dstValid[0])
      return op->emitOpError() << "expects " << fullName
                               << " valid_shape[0] to equal dst valid_shape[0]";
    if (fullValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        fullValid[1] != dstValid[1])
      return op->emitOpError() << "expects " << fullName
                               << " valid_shape[1] to equal dst valid_shape[1]";
    return checkBroadcastOperand(broadcastTy, broadcastValid, broadcastName,
                                 /*requireNonRowMajor=*/hasTmp &&
                                     targetArch == PTOArch::A3);
  };

  // (A5 tmp-form invariant is checked earlier, before the empty-marker accept.)

  if (src0MatchesDst) {
    if (succeeded(checkFullAndBroadcast(src0Ty, src0Valid, "src0", src1Ty,
                                        src1Valid, "src1")))
      return success();
  }
  if (src1MatchesDst) {
    if (succeeded(checkFullAndBroadcast(src1Ty, src1Valid, "src1", src0Ty,
                                        src0Valid, "src0")))
      return success();
  }

  return op->emitOpError() << "expects one of src0/src1 to match dst valid_shape"
                           << " and the other to be a per-row scalar vector";
}

mlir::LogicalResult mlir::pto::TRowExpandExpdifOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTRowReductionNoTmpCommon(*this, getSrc().getType(),
                                          getDst().getType(),
                                          "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowArgMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTRowArgReductionCommon(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}


mlir::LogicalResult mlir::pto::TRowMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowArgMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTRowArgReductionCommon(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}


mlir::LogicalResult mlir::pto::TRowSumOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTRowReductionNoTmpCommon(*this, getSrc().getType(),
                                          getDst().getType(),
                                          "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowProdOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A2/A3 trowprod element type to be i16/i32/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A5 trowprod element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRsqrtOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst")))
    return failure();
  auto ft = mlir::dyn_cast<mlir::FloatType>(getElemTy(ts));
  if (!ft || (!ft.isF16() && !ft.isF32()))
    return emitOpError("expects element type to be f16 or f32");
  if (getPrecisionType() == pto::RsqrtPrecision::HighPrecision && !getTmp())
    return emitOpError("expects tmp when precisionType is high_precision");
  if (auto tmp = getTmp()) {
    Type tt = tmp.getType();
    if (failed(verifyVecTileCommon(*this, tt, "tmp")))
      return failure();

    auto tmpElemTy = getElemTy(tt);
    auto tmpElemBytes = getElemBytes(tmpElemTy);
    auto tmpNumel = getStaticNumElements(getShapeVec(tt));
    if (!tmpElemBytes.has_value() || !tmpNumel.has_value())
      return emitOpError("expects tmp to have a static, byte-addressable tile type");
    if (tmpElemBytes.value() * tmpNumel.value() < 32)
      return emitOpError("expects tmp to be at least 32 bytes when provided");
  }
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TScatterOp::verify() {
  const bool hasIndexes = static_cast<bool>(getIndexes());
  const bool hasMaskPattern = static_cast<bool>(getMaskPatternAttr());
  if (hasIndexes == hasMaskPattern) {
    return emitOpError(
        "expects exactly one of indexes operand or maskPattern attribute");
  }

  auto isAllowedDataElem = [&](mlir::Type t) -> bool {
    if (t.isF16() || t.isF32() || t.isBF16()) return true;
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t))
      return (it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32);
    return false;
  };
  auto isAllowedIndexElem = [&](mlir::Type t) -> bool {
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t))
      return (it.getWidth() == 16 || it.getWidth() == 32);
    return false;
  };
  auto getMaskScatterTimes = [&](mlir::pto::MaskPatternAttr mp) -> unsigned {
    switch (mp.getValue()) {
    case mlir::pto::MaskPattern::P1111:
      return 1;
    case mlir::pto::MaskPattern::P0101:
    case mlir::pto::MaskPattern::P1010:
      return 2;
    default:
      return 4;
    }
  };

  auto verifyIndexedForm = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type ti = getIndexes().getType();
    Type td = getDst().getType();
    if (failed(verifyVecTileStorage(*this, ts, "src")) ||
        failed(verifyVecTileStorage(*this, ti, "indexes")) ||
        failed(verifyVecTileStorage(*this, td, "dst")))
      return failure();

    Type srcElem = getElemTy(ts), dstElem = getElemTy(td), idxElem = getElemTy(ti);
    if (!srcElem || !dstElem || !idxElem)
      return emitOpError("failed to get element type for operands");
    if (srcElem != dstElem)
      return emitOpError("expects src/dst to have the same element type");

    if (!isAllowedDataElem(srcElem))
      return emitOpError("expects src/dst element type to be i8/i16/i32/f16/bf16/f32");
    if (!isAllowedIndexElem(idxElem))
      return emitOpError("expects indexes element type to be i16/i32");

    auto bwData = getPTOStorageElemBitWidth(srcElem);
    auto bwIdx  = getPTOStorageElemBitWidth(idxElem);
    if (bwData != 8 && bwData != 16 && bwData != 32)
      return emitOpError("unexpected src/dst element bitwidth");

    unsigned dataBytes = bwData / 8;
    unsigned idxBytes  = bwIdx / 8;
    unsigned expectedIdxBytes = (dataBytes == 1) ? 2 : dataBytes;
    if (idxBytes != expectedIdxBytes)
      return emitOpError("expects indexes element size to match the documented scatter rule");
    return mlir::success();
  };

  auto verifyMaskForm = [&]() -> LogicalResult {
    Type ts = getSrc().getType();
    Type td = getDst().getType();
    if (failed(verifyVecTileCommon(*this, ts, "src")) ||
        failed(verifyVecTileCommon(*this, td, "dst")))
      return failure();

    auto srcTB = dyn_cast<pto::TileBufType>(ts);
    auto dstTB = dyn_cast<pto::TileBufType>(td);
    if (!srcTB || !dstTB)
      return emitOpError("expects src and dst to be tile_buf types");

    if (getElemTy(ts) != getElemTy(td))
      return emitOpError("expects src and dst to have the same element type");
    if (!isAllowedDataElem(getElemTy(ts)))
      return emitOpError("expects src/dst element type to be i8/i16/i32/f16/bf16/f32");

    auto srcValid = getValidShapeVec(ts);
    auto dstValid = getValidShapeVec(td);
    if (srcValid.size() != 2 || dstValid.size() != 2)
      return emitOpError("expects src and dst to have rank-2 valid_shape");

    auto mp = getMaskPatternAttr();
    if (!mp)
      return emitOpError("expects mask-pattern tscatter to provide maskPattern");
    const unsigned times = getMaskScatterTimes(mp);
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0])
      return emitOpError("expects src and dst to have the same valid rows");
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        srcValid[1] != static_cast<int64_t>(dstValid[1] * times))
      return emitOpError("expects src valid cols to equal dst valid cols times the mask expansion factor");

    if (srcTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        dstTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
      return emitOpError("expects mask-pattern tscatter to use row_major blayout");
    return mlir::success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (hasMaskPattern)
      return verifyMaskForm();
    return verifyIndexedForm();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (hasMaskPattern)
      return verifyMaskForm();
    return verifyIndexedForm();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TSelOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst")))
      return failure();

    Type srcElem = getElemTy(t0);
    Type src1Elem = getElemTy(t1);
    Type dstElem = getElemTy(td);
    if (!srcElem || !src1Elem || !dstElem) {
      emitOpError("failed to get element type for operands");
      return failure();
    }
    if (srcElem != src1Elem || srcElem != dstElem) {
      emitOpError("expects src0, src1, and dst to have the same element type");
      return failure();
    }

    if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) ||
        !isRowMajorTileBuf(td)) {
      emitOpError(
          "expects src0, src1, and dst to use row-major layout");
      return failure();
    }
    return srcElem;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> srcElem = verifyCommon();
    if (failed(srcElem))
      return failure();
    Type elem = *srcElem;
    bool ok = elem.isF16() || elem.isBF16() || elem.isF32();
    if (auto it = dyn_cast<IntegerType>(elem))
      ok = it.getWidth() == 16 || it.getWidth() == 32;
    if (!ok)
      return emitOpError(
          "expects A2/A3 tsel src0, src1, and dst element type to be i16/i32/f16/bf16/f32");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> srcElem = verifyCommon();
    if (failed(srcElem))
      return failure();
    Type elem = *srcElem;
    bool ok = elem.isF16() || elem.isBF16() || elem.isF32();
    if (auto it = dyn_cast<IntegerType>(elem))
      ok = it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32;
    if (!ok)
      return emitOpError(
          "expects A5 tsel src0, src1, and dst element type to be i8/i16/i32/f16/bf16/f32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TSelSOp::verify() {
  // Constraints & Verification per PTO_IR_manual.md pto.tsels:
  // - src and dst same element type; A2A3: i16/i32/f16/f32; A5: i8/i16/i32/f16/f32
  // - src and dst row-major; src and dst same valid region
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type tMask = getMask().getType();
    Type tSrc = getSrc().getType();
    Type tTmp = getTmp().getType();
    Type tDst = getDst().getType();
    if (failed(verifyTileBufCommon(*this, tMask, "mask")) ||
        failed(verifyTileBufCommon(*this, tSrc, "src")) ||
        failed(verifyTileBufCommon(*this, tTmp, "tmp")) ||
        failed(verifyTileBufCommon(*this, tDst, "dst")))
      return failure();
    Type eMask = getElemTy(tMask), eSrc = getElemTy(tSrc);
    Type eTmp = getElemTy(tTmp), eDst = getElemTy(tDst);
    if (!eMask || !eSrc || !eTmp || !eDst) {
      emitOpError("failed to get element type for operands");
      return failure();
    }
    if (eSrc != eDst)
      return emitOpError("expects src and dst to have the same element type");
    if (failed(verifyTileBufSameValidShape(*this, tSrc, tDst, "src", "dst")))
      return failure();
    return eDst;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    Type tSrc = getSrc().getType();
    Type tDst = getDst().getType();
    if (!isRowMajorTileBuf(tSrc) || !isRowMajorTileBuf(tDst))
      return emitOpError("expects src and dst to use row-major layout");
    Type elem = *elemOr;
    bool ok = elem.isF16() || elem.isF32();
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem))
      ok = (it.getWidth() == 16 || it.getWidth() == 32);
    if (!ok)
      return emitOpError(
          "expects A2/A3 tsels src and dst element type to be i16, i32, f16, or f32");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr))
      return failure();
    Type tMask = getMask().getType();
    Type tSrc = getSrc().getType();
    Type tDst = getDst().getType();
    if (!isRowMajorTileBuf(tMask) || !isRowMajorTileBuf(tSrc) || !isRowMajorTileBuf(tDst))
      return emitOpError("expects mask, src, and dst to use row-major layout");
    Type elem = *elemOr;
    bool ok = elem.isF16() || elem.isF32();
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem))
      ok = (it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32);
    if (!ok)
      return emitOpError(
          "expects A5 tsels src and dst element type to be i8, i16, i32, f16, or f32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TShlOp::verify() {
  auto verify = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyShiftLikeBinaryTileOpCommon(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects tshl src0 and src1 element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verify, verify);
}


mlir::LogicalResult mlir::pto::TShrOp::verify() {
  auto verify = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyShiftLikeBinaryTileOpCommon(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects tshr src0 and src1 element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verify, verify);
}


mlir::LogicalResult mlir::pto::TSort32Op::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type idxTy = getIdx().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst")) ||
      failed(verifyVecTileCommon(*this, idxTy, "idx")))
    return failure();
  if (getTmp() &&
      failed(verifyVecTileCommon(*this, getTmp().getType(), "tmp")))
    return failure();

  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem || srcElem != dstElem)
    return emitOpError() << "expects src and dst to have the same element type";
  if (!(srcElem.isF16() || srcElem.isF32()))
    return emitOpError() << "expects src and dst element type to be f16 or f32";

  auto idxElem = getElemTy(idxTy);
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != 32)
    return emitOpError() << "expects idx element type to be i32/u32";
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSqrtOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false)))
    return failure();
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst")))
    return failure();

  auto srcElem = getElemTy(srcTy);
  if (!(mlir::isa<mlir::FloatType>(srcElem) || mlir::isa<mlir::Float16Type>(srcElem)))
    return emitOpError() << "expects src and dst element type to be float or half";

  return mlir::success();
}



mlir::LogicalResult mlir::pto::TStoreFPOp::verify() {
  auto shouldBypassDecoded = [&]() -> bool {
    Value src = getSrc();
    Value fp = getFp();
    return isa<MemRefType>(src.getType()) || isa<MemRefType>(fp.getType()) ||
           src.getDefiningOp<pto::BindTileOp>() ||
           fp.getDefiningOp<pto::BindTileOp>();
  };

  auto verifyDstType = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (!isa<MemRefType, pto::PartitionTensorViewType>(dstTy))
      return emitOpError()
             << "expects dst to be a memref or !pto.partition_tensor_view";
    if (auto dstPart = dyn_cast<pto::PartitionTensorViewType>(dstTy)) {
      for (auto [idx, dim] : llvm::enumerate(dstPart.getShape())) {
        if (dim != ShapedType::kDynamic && dim <= 0)
          return emitOpError()
                 << "expects dst shape[" << idx << "] to be positive";
      }
    }
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type fpTy = getFp().getType();
    if (!isa<pto::TileBufType>(srcTy))
      return emitOpError() << "expects src to be a !pto.tile_buf";
    if (!isa<pto::TileBufType>(fpTy))
      return emitOpError() << "expects fp to be a !pto.tile_buf";
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")))
      return failure();
    if (failed(verifyDstType()))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::ACC)
      return emitOpError() << "expects src to be in the acc address space";
    auto srcElemTy = getElemTy(srcTy);
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() ||
          (srcIntTy && srcIntTy.getWidth() == 32)))
      return emitOpError()
             << "expects src to have element type f32, i32";
    auto srcShape = getShapeVec(srcTy);
    if (srcShape.size() != 2)
      return emitOpError() << "expects src to have rank 2";
    if (srcShape[1] != ShapedType::kDynamic &&
        (srcShape[1] < 1 || srcShape[1] > 4095))
      return emitOpError() << "expects src.cols to be in the range [1, 4095]";
    auto srcValid = getValidShapeVec(srcTy);
    if (srcValid.size() != 2)
      return emitOpError() << "expects src to have a rank-2 valid_shape";
    if (srcValid[1] != ShapedType::kDynamic &&
        (srcValid[1] < 0 || srcValid[1] > 4095))
      return emitOpError()
             << "expects src.valid_shape[1] to be in the range [0, 4095]";
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type fpTy = getFp().getType();
    if (!isa<pto::TileBufType>(srcTy))
      return emitOpError() << "expects src to be a !pto.tile_buf";
    if (!isa<pto::TileBufType>(fpTy))
      return emitOpError() << "expects fp to be a !pto.tile_buf";
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, fpTy, "fp")))
      return failure();
    if (failed(verifyDstType()))
      return failure();
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::ACC)
      return emitOpError() << "expects src to be in the acc address space";
    return mlir::success();
  };
  if (shouldBypassDecoded())
    return success();
  switch (getVerifierTargetArch(getOperation())) {
  case VerifierTargetArch::A2A3:
    return verifyA2A3();
  case VerifierTargetArch::A5:
    return verifyA5();
  }
  return failure();
}


mlir::LogicalResult mlir::pto::TSubOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/false,
      "expects A2/A3 tsub element type to be i32/i16/f16/f32",
      "expects A5 tsub element type to be i32/i16/i8/f16/f32");
}


mlir::LogicalResult mlir::pto::TSubCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type src2Ty = getSrc2().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(src2Ty) || !isPTOShapedLike(dstTy))
    return emitOpError() << "expects PTO shaped-like src0, src1, src2, and dst";

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size() || getShapeVec(src2Ty).size() != d.size())
    return emitOpError() << "expects all tensors to have the same rank";
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSubSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tsubs element type to be i32/i16/f16/f32",
      "expects A5 tsubs element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}


mlir::LogicalResult mlir::pto::TSubSCOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(dstTy))
    return emitOpError() << "expects PTO shaped-like src0, src1, and dst";

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size())
    return emitOpError() << "expects src0, src1, and dst to have the same rank";
  return mlir::success();
}
mlir::LogicalResult mlir::pto::TTransOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type tmpTy = getTmp().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, tmpTy, "tmp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    Type srcElem = getElemTy(srcTy);
    Type tmpElem = getElemTy(tmpTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !tmpElem || !dstElem || srcElem != dstElem || srcElem != tmpElem)
      return emitOpError() << "expects src and dst to have the same element type";
    if (auto srcTb = dyn_cast<pto::TileBufType>(srcTy)) {
      if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor))
        return emitOpError() << "expects A2/A3 transpose src to use the row_major blayout";
    }
    unsigned elemBytes = getPTOStorageElemByteSize(srcElem);
    if (elemBytes == 0)
      return emitOpError() << "failed to get transpose element size";
    if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4)
      return emitOpError() << "expects transpose element size to be 1, 2, or 4 bytes";
    auto isAllowedWidthType = [&](Type ty) {
      if (elemBytes == 4)
        return ty.isInteger(32) || ty.isF32();
      if (elemBytes == 2)
        return ty.isInteger(16) || ty.isF16() || ty.isBF16();
      return ty.isInteger(8);
    };
    if (!isAllowedWidthType(srcElem))
      return emitOpError() << "expects transpose element type to match the supported set for its width";
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type tmpTy = getTmp().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, tmpTy, "tmp")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst")))
      return failure();
    Type srcElem = getElemTy(srcTy);
    Type tmpElem = getElemTy(tmpTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !tmpElem || !dstElem || srcElem != dstElem || srcElem != tmpElem)
      return emitOpError() << "expects src, tmp, and dst to have the same element type";
    unsigned elemBytes = getPTOStorageElemByteSize(srcElem);
    if (elemBytes == 0)
      return emitOpError() << "failed to get transpose element size";
    if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4)
      return emitOpError() << "expects transpose element size to be 1, 2, or 4 bytes";
    auto isAllowedWidthType = [&](Type ty) {
      if (elemBytes == 4)
        return ty.isInteger(32) || ty.isF32();
      if (elemBytes == 2)
        return ty.isInteger(16) || ty.isF16() || ty.isBF16();
      return ty.isInteger(8);
    };
    if (!isAllowedWidthType(srcElem))
      return emitOpError() << "expects transpose element type to match the supported set for its width";
    auto checkAlignedMajor = [&](Type ty, StringRef name) -> LogicalResult {
      auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
      if (!tb)
        return success();
      auto shape = getShapeVec(ty);
      if (shape.size() != 2)
        return success();
      bool rowMajor = tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
      int64_t major = rowMajor ? shape[1] : shape[0];
      if (major != ShapedType::kDynamic && (major * static_cast<int64_t>(elemBytes)) % 32 != 0)
        return emitOpError() << "expects " << name << " major dimension times element size to be 32-byte aligned on A5";
      return success();
    };
    if (failed(checkAlignedMajor(srcTy, "src")) || failed(checkAlignedMajor(dstTy, "dst")))
      return failure();
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TXorOp::verify() {
  auto verifyBase = [&]() -> FailureOr<Type> {
    return verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyBase();
    if (failed(elemOr))
      return failure();
    Type tmpTy = getTmp().getType();
    if (failed(verifyTileBufCommon(*this, tmpTy, "tmp")))
      return failure();
    Type elem = *elemOr;
    if (getElemTy(tmpTy) != elem)
      return emitOpError("expects tmp to have the same element type as src0, src1, and dst");
    if (!isRowMajorTileBuf(tmpTy))
      return emitOpError("expects tmp to use row-major layout");
    if (failed(verifyTileBufSameValidShape(*this, tmpTy, getDst().getType(), "tmp", "dst")))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(elem);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16))
      return emitOpError(
          "expects A2/A3 txor src0, src1, tmp, and dst element type to be i8/i16");
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyBase();
    if (failed(elemOr))
      return failure();
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32))
      return emitOpError(
          "expects A5 txor src0, src1, and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TXorSOp::verify() {
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
          "expects A2/A3 txors src and dst element type to be i8/i16");
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
          "expects A5 txors src and dst element type to be i8/i16/i32");
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
mlir::LogicalResult mlir::pto::TPrintOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto srcType = getSrc().getType();
  if (auto tb = mlir::dyn_cast<mlir::pto::TileBufType>(srcType)) {
    auto elem = tb.getElementType();
    if (!(elem.isF16() || elem.isF32() ||
          elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32)))
      return emitOpError() << "expects printable tile element type";
    auto space = getPTOMemorySpaceEnum(srcType);
    if (!space || *space != pto::AddressSpace::VEC)
      return emitOpError() << "expects printable tile_buf to be in vec address space";
    return success();
  }
  if (mlir::dyn_cast<MemRefType>(srcType) ||
      mlir::dyn_cast<mlir::pto::PartitionTensorViewType>(srcType))
    return mlir::success();
  return emitOpError() << "expects tile_buf, memref, or partition_tensor_view for src";
}



[[maybe_unused]] static LogicalResult verifyMatmulCommon(Operation *op, Value lhs, Value rhs,
                                       Value biasOpt, Type maybeDstElemTy,
                                       Type maybeResultElemTy) {
  // ---- case A: tensor/memref (ShapedType) ----
  if (auto lhsTy = dyn_cast<ShapedType>(lhs.getType())) {
    auto rhsTy = dyn_cast<ShapedType>(rhs.getType());
    if (!rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank())
      return op->emitOpError("expects lhs and rhs to be ranked tensors or memrefs");

    if (lhsTy.getElementType() != rhsTy.getElementType())
      return op->emitOpError()
             << "expects lhs and rhs to have the same element type, but got lhs="
             << lhsTy.getElementType() << " rhs=" << rhsTy.getElementType();

    if (biasOpt) {
      auto biasTy = dyn_cast<ShapedType>(biasOpt.getType());
      if (!biasTy || !biasTy.hasRank())
        return op->emitOpError("expects bias to be a ranked tensor or memref");
      if (biasTy.getElementType() != lhsTy.getElementType())
        return op->emitOpError()
               << "expects bias to have the same element type as lhs and rhs, but got bias="
               << biasTy.getElementType() << " vs " << lhsTy.getElementType();
    }

    if (maybeDstElemTy && maybeDstElemTy != lhsTy.getElementType())
      return op->emitOpError()
             << "expects dst to have the same element type as lhs and rhs, but got dst="
             << maybeDstElemTy << " vs " << lhsTy.getElementType();

    if (maybeResultElemTy && maybeResultElemTy != lhsTy.getElementType())
      return op->emitOpError()
             << "expects result to have the same element type as lhs and rhs, but got result="
             << maybeResultElemTy << " vs " << lhsTy.getElementType();

    return success();
  }

  // ---- case B: tile ----
  auto lhsTile = dyn_cast<mlir::pto::TileType>(lhs.getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(rhs.getType());
  if (!lhsTile || !rhsTile)
    return op->emitOpError("expects lhs and rhs to be ranked tensors, memrefs, or !pto.tile");

  if (lhsTile.getElementType() != rhsTile.getElementType())
    return op->emitOpError() << "expects lhs and rhs tiles to have the same element type, but got lhs="
                             << lhsTile.getElementType() << " rhs=" << rhsTile.getElementType();

  if ((int64_t)lhsTile.getShape().size() != 2 || (int64_t)rhsTile.getShape().size() != 2)
    return op->emitOpError("expects lhs and rhs tiles to be 2D");

  if (lhsTile.getShape()[1] != rhsTile.getShape()[0])
    return op->emitOpError() << "expects lhs dim1 to equal rhs dim0, but got "
                             << lhsTile.getShape()[1] << " vs " << rhsTile.getShape()[0];

  if (biasOpt) {
    auto biasTile = dyn_cast<mlir::pto::TileType>(biasOpt.getType());
    if (!biasTile)
      return op->emitOpError("expects bias to be !pto.tile when lhs and rhs are !pto.tile");
    if (biasTile.getElementType() != lhsTile.getElementType())
      return op->emitOpError("expects bias to have the same element type as lhs and rhs");
  }

  if (maybeDstElemTy && maybeDstElemTy != lhsTile.getElementType())
    return op->emitOpError() << "expects dst to have the same element type as lhs and rhs";

  if (maybeResultElemTy && maybeResultElemTy != lhsTile.getElementType())
    return op->emitOpError() << "expects result to have the same element type as lhs and rhs";

  return success();
}

LogicalResult mlir::pto::TMatmulOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType())))
      return failure();
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    return verifyMatmulLike(*this, getLhs().getType(), getRhs().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TGemvOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyGemvTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                      getDst().getType())))
      return failure();
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType()))))
      return failure();
    return verifyMatmulLike(*this, getLhs().getType(), getRhs().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TMatmulAccOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
      failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                   getDst().getType())))
    return failure();
  return success();
}

LogicalResult mlir::pto::TGemvAccOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
      failed(verifyGemvTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                    getDst().getType())))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// inferReturnTypes() for matmul ops (keep your existing code)
//===----------------------------------------------------------------------===
[[maybe_unused]] static mlir::Type inferMatmulTileResult2DFromAB(MLIRContext *context, ValueRange operands) {
  if (operands.size() < 2)
    return mlir::Type();

  auto lhsTile = dyn_cast<mlir::pto::TileType>(operands[0].getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(operands[1].getType());
  if (!lhsTile || !rhsTile)
    return mlir::Type();

  Type elemTy = lhsTile.getElementType();

  if (operands.size() >= 3) {
    if (auto biasTile = dyn_cast<mlir::pto::TileType>(operands[2].getType())) {
      return mlir::pto::TileType::get(context, biasTile.getShape(), elemTy);
    }
  }

  auto lhsShape = lhsTile.getShape();
  auto rhsShape = rhsTile.getShape();
  if (lhsShape.size() >= 2 && rhsShape.size() >= 2) {
    int64_t M = lhsShape[0];
    int64_t N = rhsShape[1];
    llvm::SmallVector<int64_t, 2> outShape = {M, N};
    return mlir::pto::TileType::get(context, outShape, elemTy);
  }

  return mlir::Type();
}

[[maybe_unused]] static RankedTensorType inferMatmulResult2DFromAB(ValueRange operands) {
  if (operands.size() < 2)
    return RankedTensorType();

  auto lhsTy = dyn_cast<ShapedType>(operands[0].getType());
  auto rhsTy = dyn_cast<ShapedType>(operands[1].getType());
  if (!lhsTy || !rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank())
    return RankedTensorType();

  Type elemTy = lhsTy.getElementType();

  if (operands.size() >= 3) {
    if (auto biasRT = dyn_cast<RankedTensorType>(operands[2].getType()))
      return RankedTensorType::get(biasRT.getShape(), elemTy);
    if (auto biasMR = dyn_cast<MemRefType>(operands[2].getType())) {
      if (biasMR.hasStaticShape())
        return RankedTensorType::get(biasMR.getShape(), elemTy);
    }
  }

  if (lhsTy.getRank() >= 2 && rhsTy.getRank() >= 2) {
    int64_t M = lhsTy.getDimSize(0);
    int64_t N = rhsTy.getDimSize(1);
    return RankedTensorType::get({M, N}, elemTy);
  }

  return RankedTensorType();
}

[[maybe_unused]] static RankedTensorType inferAccReturnFromAccIn(ValueRange operands) {
  if (operands.empty())
    return RankedTensorType();
  if (auto accRT = dyn_cast<RankedTensorType>(operands[0].getType()))
    return accRT;
  return RankedTensorType();
}

namespace mlir {
namespace pto {

static LogicalResult parseShapeAndElem(AsmParser &parser,
                                       SmallVectorImpl<int64_t> &shape,
                                       Type &elementType,
                                       bool allowDynamic) {
  if (parser.parseLess())
    return failure();

  if (parser.parseDimensionList(shape, allowDynamic))
    return failure();

  if (parser.parseType(elementType))
    return failure();

  if (parser.parseGreater())
    return failure();

  return success();
}

static void printShapeAndElem(AsmPrinter &printer,
                              ArrayRef<int64_t> shape,
                              Type elementType) {
  printer << "<";
  for (auto d : shape) {
    if (d == ShapedType::kDynamic)
      printer << "?";
    else
      printer << d;
    printer << "x";
  }
  printer.printType(elementType);
  printer << ">";
}

// =============================================================================
// PartitionTensorViewType Implementation
// =============================================================================

Type PartitionTensorViewType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true)))
    return Type();
  
  return PartitionTensorViewType::get(parser.getContext(), shape, elemTy);
}

void PartitionTensorViewType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

// ---- TileType ----
Type TileType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true)))
    return Type();
  return TileType::get(parser.getContext(), shape, elemTy);
}

void TileType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

// ---- LocalArrayType ----
// Asm form: !pto.local_array<D1 x D2 x ... x Dk x T>
// Static shape only (no '?'). Element type must be a scalar; this is enforced
// by the type verifier below.
Type LocalArrayType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/false)))
    return Type();
  return LocalArrayType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), shape, elemTy);
}

void LocalArrayType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

LogicalResult LocalArrayType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, Type elementType) {
  if (shape.empty())
    return emitError() << "'!pto.local_array' requires at least one dimension";
  for (auto [i, d] : llvm::enumerate(shape)) {
    if (d <= 0)
      return emitError()
             << "'!pto.local_array' dimension " << i
             << " must be a positive static size, got " << d;
  }
  if (!elementType.isIntOrFloat())
    return emitError()
           << "'!pto.local_array' element type must be a scalar integer or "
              "float, got "
           << elementType;
  return success();
}

// =============================================================================
// Decompose Helper (Reverse Engineering AffineMap -> Strides)
// =============================================================================

// Helper: 递归地将 Add 表达式拆解为单独的项列表
static void flattenAddExpr(AffineExpr expr, SmallVectorImpl<AffineExpr> &terms) {
  if (auto add = llvm::dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (add.getKind() == AffineExprKind::Add) {
      flattenAddExpr(add.getLHS(), terms);
      flattenAddExpr(add.getRHS(), terms);
      return;
    }
  }
  terms.push_back(expr);
}

// Helper: 从 AffineMap 中提取 Strides
static void decomposeStridedLayout(AffineMap map, SmallVectorImpl<int64_t> &strides) {
  // 1. 初始化
  strides.assign(map.getNumDims(), 0);
  
  if (map.getNumResults() != 1) return;
  
  // 2. 摊平表达式
  SmallVector<AffineExpr, 4> terms;
  flattenAddExpr(map.getResult(0), terms);

  // 3. 分析每一项
  for (auto term : terms) {
    // 情况 A: dN * Const 或 Const * dN
    if (auto mul = llvm::dyn_cast<AffineBinaryOpExpr>(term)) {
      if (mul.getKind() == AffineExprKind::Mul) {
        AffineExpr lhs = mul.getLHS();
        AffineExpr rhs = mul.getRHS();

        // 尝试匹配 LHS=Dim, RHS=Const
        if (auto dim = llvm::dyn_cast<AffineDimExpr>(lhs)) {
          if (auto cst = llvm::dyn_cast<AffineConstantExpr>(rhs)) {
            strides[dim.getPosition()] = cst.getValue();
            continue;
          }
        }
        
        // 尝试匹配 LHS=Const, RHS=Dim (乘法交换律)
        if (auto dim = llvm::dyn_cast<AffineDimExpr>(rhs)) {
          if (auto cst = llvm::dyn_cast<AffineConstantExpr>(lhs)) {
            strides[dim.getPosition()] = cst.getValue();
            continue;
          }
        }
      }
    }
    // 情况 B: 单独的 dN (隐含 Stride = 1)
    else if (auto dim = llvm::dyn_cast<AffineDimExpr>(term)) {
      strides[dim.getPosition()] = 1;
    }
  }
}

// =============================================================================
// [Critical] Strict Alignment Protocol Helper
// =============================================================================
// This function is the SINGLE source of truth for building the AffineMap.
// Both the Parser and the Op Inference MUST use this exact function.
// It ensures that the order of AffineExpr addition is:
//   0 + (d0*str0 + d1*str1...) + (s0*str0 + s1*str1...)
// This guarantees bitwise-identical AffineMaps for verification.
static AffineMap buildStrictBitwiseAffineMap(MLIRContext *ctx, 
                                             ArrayRef<int64_t> strides, 
                                             bool isMultiDimSymbol) {
  unsigned rank = strides.size();
  
  // Step 1: Initialize with Constant(0)
  AffineExpr totalExpr = getAffineConstantExpr(0, ctx);

  // Step 2: Add Dimensions (d0*str0 + d1*str1...)
  // Strictly in order: 0, 1, 2...
  for (unsigned i = 0; i < rank; ++i) {
    auto dim = getAffineDimExpr(i, ctx);
    auto str = getAffineConstantExpr(strides[i], ctx);
    totalExpr = totalExpr + (dim * str);
  }

  // Step 3: Add Symbols (s0*str0 + s1*str1...)
  // Strictly in order: 0, 1, 2...
  if (isMultiDimSymbol) {
    for (unsigned i = 0; i < rank; ++i) {
      auto sym = getAffineSymbolExpr(i, ctx);
      auto str = getAffineConstantExpr(strides[i], ctx);
      totalExpr = totalExpr + (sym * str);
    }
  } 
  // (Optional: handle single dynamic offset case if needed, omitted for clarity)

  // numSymbols is rank if multi-dim (for offsets), else 0
  unsigned numSymbols = isMultiDimSymbol ? rank : 0;
  return AffineMap::get(rank, numSymbols, totalExpr);
}


// =============================================================================
// Parser Implementation
// =============================================================================

// Helper for parsing [64, 1]
static ParseResult parseStrideList(AsmParser &parser, SmallVectorImpl<int64_t> &strides) {
  if (parser.parseLSquare()) return failure();
  do {
    int64_t stride;
    if (parser.parseInteger(stride)) return failure();
    strides.push_back(stride);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRSquare()) return failure();
  return success();
}

// The custom attribute parser for: strided<[64, 1], offset: [?, ?]>
[[maybe_unused]] static ParseResult parseStridedLayout(AsmParser &parser, Attribute &layout) {
  if (parser.parseLess()) return failure();
  
  // 1. Parse Strides
  SmallVector<int64_t> strides;
  if (parseStrideList(parser, strides)) return failure();
  
  bool isMultiDim = false;
  unsigned numSymbols = 0;

  // 2. Parse Offset
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("offset") || parser.parseColon()) return failure();
    
    // Check for multi-dim syntax: [?, ?]
    if (succeeded(parser.parseOptionalLSquare())) {
      isMultiDim = true;
      do {
        if (parser.parseQuestion()) return failure();
        numSymbols++;
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRSquare()) return failure();
    } else {
      // Fallback for old scalar syntax '?'
      if (parser.parseOptionalQuestion()) { /* handle single scalar */ }
    }
  }
  
  if (parser.parseGreater()) return failure();

  // 3. Validation
  if (isMultiDim && numSymbols != strides.size()) {
    return parser.emitError(parser.getCurrentLocation(), 
                            "Number of offset symbols must match rank");
  }

  // 4. [CALL SHARED BUILDER]
  // Delegate to the strict builder
  MLIRContext *ctx = parser.getContext();
  AffineMap map = buildStrictBitwiseAffineMap(ctx, strides, isMultiDim);
  
  layout = AffineMapAttr::get(map);
  return success();
}

// =============================================================================
// Printer Implementation
// =============================================================================

[[maybe_unused]] static void printLayout(AsmPrinter &printer, Attribute layoutAttr) {
  if (!layoutAttr) return;
  auto mapAttr = llvm::dyn_cast<AffineMapAttr>(layoutAttr);
  if (!mapAttr) { printer << ", " << layoutAttr; return; }

  AffineMap map = mapAttr.getValue();
  if (map.isIdentity()) return; 

  // 1. [核心修改] 反解 Strides
  SmallVector<int64_t> strides;
  decomposeStridedLayout(map, strides);

  printer << ", strided<[";
  // 2. 打印真实的 strides
  llvm::interleaveComma(strides, printer); 
  printer << "]";

  // Print Offset: [?, ?]
  unsigned numSyms = map.getNumSymbols();
  if (numSyms > 0) {
    printer << ", offset: [";
    for (unsigned i = 0; i < numSyms; ++i) {
      printer << "?";
      if (i < numSyms - 1) printer << ", ";
    }
    printer << "]";
  }
  printer << ">";
}

// ---- TileBuf ---


// Tile subview 相关实现

// =============================================================================
// Op Interface Implementation: SubViewOp
// =============================================================================

ParseResult mlir::pto::SubViewOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> valids;
  Type sourceTy;
  Type resultTy;
  bool hasExplicitResultTy = false;

  if (parser.parseOperand(source) || parser.parseLSquare() ||
      parser.parseOperandList(offsets) || parser.parseRSquare() ||
      parser.parseKeyword("sizes"))
    return failure();

  ArrayAttr sizesAttr;
  if (parser.parseAttribute(sizesAttr, "sizes", result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("valid"))) {
    OpAsmParser::UnresolvedOperand vrow, vcol;
    if (parser.parseLSquare() || parser.parseOperand(vrow) || parser.parseComma() ||
        parser.parseOperand(vcol) || parser.parseRSquare())
      return failure();
    valids.push_back(vrow);
    valids.push_back(vcol);
  }

  if (parser.parseOptionalAttrDict(result.attributes) ||
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
  if (parser.resolveOperands(offsets, indexTy, result.operands))
    return failure();
  if (!valids.empty() &&
      parser.resolveOperands(valids, indexTy, result.operands))
    return failure();

  int32_t hasValid = valids.empty() ? 0 : 1;
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(offsets.size()), hasValid, hasValid}));

  if (hasExplicitResultTy) {
    result.addTypes(resultTy);
    return success();
  }

  SmallVector<Type> inferredReturnTypes;
  DictionaryAttr attrs = result.attributes.getDictionary(parser.getContext());
  if (failed(SubViewOp::inferReturnTypes(
          parser.getContext(), std::nullopt, result.operands, attrs, nullptr,
          RegionRange(), inferredReturnTypes))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "failed to infer pto.subview result type");
  }
  result.addTypes(inferredReturnTypes);
  return success();
}

void mlir::pto::SubViewOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << "[";
  printer.printOperands(getOffsets());
  printer << "] sizes " << getSizes();
  if (getValidRow()) {
    printer << " valid [" << getValidRow() << ", " << getValidCol() << "]";
  }
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes",
                                                 "sizes"});
  printer << " : " << getSource().getType() << " -> " << getResult().getType();
}

// The inferred result type derives valid_shape from `sizes` (or the explicit
// valid operands). With the operand omitted the result type is authoritative for
// the valid extent (any static value, including the v=0 no-op-replay marker or a
// partial valid), so accept a static declared valid that differs from the
// size-inferred one here; SubViewOp::verify() enforces the precise per-path rule
// (operand clamping vs the [0, size] range). Only a dynamic declared valid that
// disagrees with the inferred extent is incompatible -- it needs an explicit
// operand to supply the runtime value. Every other difference (shape, element
// type, address space, config) is still rejected as the default check would.
bool SubViewOp::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [inferred, declared] : llvm::zip(lhs, rhs)) {
    if (inferred == declared)
      continue;
    auto inferredTb = dyn_cast<TileBufType>(inferred);
    auto declaredTb = dyn_cast<TileBufType>(declared);
    if (!inferredTb || !declaredTb)
      return false;
    if (inferredTb.getShape() != declaredTb.getShape() ||
        inferredTb.getElementType() != declaredTb.getElementType() ||
        inferredTb.getMemorySpace() != declaredTb.getMemorySpace() ||
        inferredTb.getConfigAttr() != declaredTb.getConfigAttr())
      return false;
    auto inferredValid = inferredTb.getValidShape();
    auto declaredValid = declaredTb.getValidShape();
    if (inferredValid.size() != declaredValid.size())
      return false;
    for (auto [inferredDim, declaredDim] : llvm::zip(inferredValid, declaredValid)) {
      // Any static declared valid extent is accepted in place of the inferred
      // one; only a dynamic declared valid that disagrees is incompatible.
      if (inferredDim != declaredDim && declaredDim == ShapedType::kDynamic)
        return false;
    }
  }
  return true;
}

LogicalResult SubViewOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  // 1. 获取 Source Type
  if (operands.empty()) return failure();
  auto sourceType = llvm::dyn_cast<TileBufType>(operands[0].getType());
  if (!sourceType) return failure();

  // 2. 获取 subview 逻辑窗口（sizes）
  ArrayAttr sizeAttr;
  if (properties) {
    const auto *prop = properties.as<SubViewOp::Properties *>();
    if (prop) sizeAttr = prop->sizes;
  }
  if (!sizeAttr && attributes) {
    sizeAttr = attributes.getAs<ArrayAttr>("sizes");
  }
  if (!sizeAttr) return failure();

  SmallVector<int64_t> subviewShape;
  for (auto attr : sizeAttr) {
    int64_t dim = llvm::cast<IntegerAttr>(attr).getInt();
    subviewShape.push_back(dim);
  }

  // Design: subview 的结果 tile 类型显式表达逻辑子窗口 shape（sizes）。
  ArrayRef<int64_t> parentShape = sourceType.getShape();
  if (subviewShape.size() != parentShape.size())
    return failure();

  // Derive valid shape from explicit valid_row/valid_col when provided.
  // Otherwise default to subview shape (no parent valid-shape inheritance).
  SmallVector<int64_t> validShape;
  constexpr int64_t kDynamicValidDim = -1;
  int64_t rank = static_cast<int64_t>(subviewShape.size());
  Value explicitVRow;
  Value explicitVCol;

  // Robustly decode optional valid operands using AttrSizedOperandSegments:
  //   [source, offsets..., valid_row?, valid_col?]
  if (attributes) {
    if (auto segAttr =
            attributes.getAs<DenseI32ArrayAttr>("operandSegmentSizes")) {
      ArrayRef<int32_t> segs = segAttr.asArrayRef();
      if (segs.size() == 4) {
        int32_t srcSeg = segs[0];
        int32_t offSeg = segs[1];
        int32_t vRowSeg = segs[2];
        int32_t vColSeg = segs[3];
        if (srcSeg == 1 && offSeg >= 0 && (vRowSeg == 0 || vRowSeg == 1) &&
            (vColSeg == 0 || vColSeg == 1)) {
          size_t idx = static_cast<size_t>(srcSeg + offSeg);
          if (vRowSeg == 1 && idx < operands.size())
            explicitVRow = operands[idx++];
          if (vColSeg == 1 && idx < operands.size())
            explicitVCol = operands[idx];
        }
      }
    }
  }

  // Fallback for legacy callers that may not provide operandSegmentSizes.
  if (!explicitVRow && !explicitVCol && rank == 2) {
    size_t expectedWithoutValid = static_cast<size_t>(1 + rank);
    if (operands.size() >= expectedWithoutValid + 2) {
      explicitVRow = operands[expectedWithoutValid];
      explicitVCol = operands[expectedWithoutValid + 1];
    }
  }

  for (size_t i = 0, e = subviewShape.size(); i < e; ++i) {
    int64_t vdim = subviewShape[i];
    Value explicitV = (i == 0) ? explicitVRow : (i == 1 ? explicitVCol : Value());
    if (explicitV) {
      auto cst = getConstIndexValue(explicitV);
      vdim = cst ? std::min<int64_t>(*cst, subviewShape[i]) : kDynamicValidDim;
    }
    validShape.push_back(vdim);
  }

  // 3. 继承 Config (若为空使用默认)
  auto cfg = sourceType.getConfigAttr();
  if (!cfg) cfg = TileBufConfigAttr::getDefault(context);

  // 4. 构建 Result Type
  auto canonicalValidShape = canonicalizeTileBufValidShape(validShape);
  auto resultType = TileBufType::get(
      context, subviewShape, sourceType.getElementType(),
      sourceType.getMemorySpace(), canonicalValidShape, cfg);

  inferredReturnTypes.push_back(resultType);
  return success();
}

// =============================================================================
// SubViewOp verifier
// =============================================================================
static bool getConstIndex(Value v, int64_t &out) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    out = cOp.value();
    return true;
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    out = cInt.value();
    return true;
  }
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue())) {
      out = ia.getInt();
      return true;
    }
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndex(castOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
    return getConstIndex(extOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
    return getConstIndex(extOp.getIn(), out);
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>())
    return getConstIndex(truncOp.getIn(), out);
  return false;
}

static LogicalResult computeInnerShape(TileBufConfigAttr cfg, Type elemTy,
                                       int64_t &innerRows, int64_t &innerCols,
                                       bool &boxed, int32_t &bl, int32_t &sl) {
  auto readBLayoutI32 = [](Attribute attr, int32_t &out) -> bool {
    if (auto a = dyn_cast<BLayoutAttr>(attr)) {
      out = (int32_t)a.getValue();
      return true;
    }
    if (auto a = dyn_cast<IntegerAttr>(attr)) {
      out = (int32_t)a.getInt();
      return true;
    }
    return false;
  };
  auto readSLayoutI32 = [](Attribute attr, int32_t &out) -> bool {
    if (auto a = dyn_cast<SLayoutAttr>(attr)) {
      out = (int32_t)a.getValue();
      return true;
    }
    if (auto a = dyn_cast<IntegerAttr>(attr)) {
      out = (int32_t)a.getInt();
      return true;
    }
    return false;
  };
  bl = 0;
  sl = 0;
  int32_t fr = 512;
  (void)readBLayoutI32(cfg.getBLayout(), bl);
  (void)readSLayoutI32(cfg.getSLayout(), sl);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize())) fr = (int32_t)attr.getInt();

  boxed = (sl != 0);
  if (!boxed) {
    innerRows = 1;
    innerCols = 1;
    return success();
  }

  int64_t elemBytes = static_cast<int64_t>(getElemByteSize(elemTy));
  if (elemBytes <= 0) return failure();

  if (fr == 1024) {
    innerRows = 16;
    innerCols = 16;
    return success();
  }
  if (fr == 32) {
    innerRows = 16;
    innerCols = 2;
    return success();
  }
  if (fr == 512) {
    if (sl == 1) {
      innerRows = 16;
      innerCols = 32 / elemBytes;
      return success();
    }
    if (sl == 2) {
      innerRows = 32 / elemBytes;
      innerCols = 16;
      return success();
    }
  }
  return failure();
}

static LogicalResult
computeExpectedTileBufMemrefStrides(TileBufType tileTy,
                                    SmallVectorImpl<int64_t> &expectedStrides) {
  if (tileTy.getRank() != 2)
    return failure();

  ArrayRef<int64_t> shape = tileTy.getShape();
  if (shape.size() != 2)
    return failure();
  if (shape[0] == ShapedType::kDynamic || shape[1] == ShapedType::kDynamic)
    return failure();

  auto cfg = tileTy.getConfigAttr();
  if (!cfg)
    cfg = TileBufConfigAttr::getDefault(tileTy.getContext());

  int64_t innerRows = 1, innerCols = 1;
  bool boxed = false;
  int32_t bl = 0, sl = 0;
  if (failed(computeInnerShape(cfg, tileTy.getElementType(), innerRows, innerCols,
                               boxed, bl, sl)))
    return failure();

  expectedStrides.clear();
  if (!boxed) {
    if (bl == 1) {
      expectedStrides.push_back(1);
      expectedStrides.push_back(shape[0]);
    } else {
      expectedStrides.push_back(shape[1]);
      expectedStrides.push_back(1);
    }
    return success();
  }

  if (bl == 1) {
    if (sl != 1)
      return failure();
    expectedStrides.push_back(innerCols);
    expectedStrides.push_back(shape[0]);
    return success();
  }

  expectedStrides.push_back(shape[1]);
  expectedStrides.push_back(innerRows);
  return success();
}

mlir::LogicalResult mlir::pto::SimdTileToMemrefOp::verify() {
  auto memTy = dyn_cast<MemRefType>(getDst().getType());
  if (!memTy)
    return emitOpError("expects result to be memref");

  Type srcTy = getSrc().getType();
  if (auto tileTy = dyn_cast<TileBufType>(srcTy)) {
    if (memTy.getElementType() != tileTy.getElementType())
      return emitOpError(
          "expects memref element type to match tile_buf element type");

    if (memTy.getMemorySpace() != tileTy.getMemorySpace())
      return emitOpError(
          "expects memref memory space to match tile_buf memory space");

    if (memTy.getRank() != tileTy.getRank())
      return emitOpError("expects memref rank to match tile_buf rank");

    ArrayRef<int64_t> tileShape = tileTy.getShape();
    ArrayRef<int64_t> validShape = tileTy.getValidShape();
    ArrayRef<int64_t> memShape = memTy.getShape();
    if (tileShape.size() != memShape.size())
      return emitOpError(
          "expects memref shape rank to match tile_buf shape rank");

    if (validShape.size() != memShape.size())
      return emitOpError(
          "expects tile_buf valid shape rank to match memref shape rank");

    for (unsigned i = 0; i < validShape.size(); ++i) {
      int64_t expect = validShape[i];
      if (expect < 0) {
        if (memShape[i] >= 0 && memShape[i] != tileShape[i]) {
          return emitOpError()
                 << "expects memref dim " << i
                 << " to be dynamic or match physical tile dim " << tileShape[i]
                 << " because tile_buf valid dim is ?";
        }
        continue;
      }

      if (memShape[i] != expect) {
        return emitOpError() << "expects memref dim " << i
                             << " to match tile_buf valid dim; got "
                             << memShape[i] << ", expected " << expect;
      }
    }

    SmallVector<int64_t, 4> expectedStrides;
    if (failed(computeExpectedTileBufMemrefStrides(tileTy, expectedStrides)))
      return emitOpError("cannot infer expected strides from tile_buf layout");

    SmallVector<int64_t, 4> memStrides;
    int64_t memOffset = ShapedType::kDynamic;
    if (failed(getStridesAndOffset(memTy, memStrides, memOffset)))
      return emitOpError("expects memref to use strided layout");
    if (memOffset != 0)
      return emitOpError("expects memref offset to be 0");
    if (memStrides.size() != expectedStrides.size())
      return emitOpError("expects memref stride rank to match tile_buf rank");
    for (unsigned i = 0; i < expectedStrides.size(); ++i) {
      if (memStrides[i] != expectedStrides[i]) {
        return emitOpError()
               << "expects memref strides to match tile_buf layout; got "
               << memStrides[i] << " at dim " << i << ", expected "
               << expectedStrides[i];
      }
    }
    return success();
  }

  auto srcMemTy = dyn_cast<MemRefType>(srcTy);
  if (!srcMemTy)
    return emitOpError("expects src to be !pto.tile_buf or memref");

  if (srcMemTy.getElementType() != memTy.getElementType())
    return emitOpError("expects src/result memref element types to match");

  if (srcMemTy.getMemorySpace() != memTy.getMemorySpace())
    return emitOpError("expects src/result memref memory spaces to match");

  if (srcMemTy.getRank() != memTy.getRank())
    return emitOpError("expects src/result memref ranks to match");

  ArrayRef<int64_t> srcShape = srcMemTy.getShape();
  ArrayRef<int64_t> dstShape = memTy.getShape();
  for (unsigned i = 0; i < srcShape.size(); ++i) {
    if (srcShape[i] >= 0 && dstShape[i] >= 0 && srcShape[i] != dstShape[i]) {
      return emitOpError()
             << "expects compatible src/result memref shapes; dim " << i
             << " mismatches (" << srcShape[i] << " vs " << dstShape[i] << ")";
    }
  }

  return success();
}

mlir::LogicalResult mlir::pto::SubViewOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects tile_buf src and tile_buf result");
  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError("expects rank-2 tilebuf for src/dst");

  auto sizesAttr = getSizes();
  if (!sizesAttr || sizesAttr.size() != 2)
    return emitOpError("subview expects 2D sizes");
  int64_t sizeR = cast<IntegerAttr>(sizesAttr[0]).getInt();
  int64_t sizeC = cast<IntegerAttr>(sizesAttr[1]).getInt();
  if (sizeR <= 0 || sizeC <= 0)
    return emitOpError("subview sizes must be positive");
  if (getOffsets().size() != 2)
    return emitOpError("subview expects 2D offsets");

  int64_t offR = 0, offC = 0;
  bool offRConst = getConstIndex(getOffsets()[0], offR);
  bool offCConst = getConstIndex(getOffsets()[1], offC);
  if (offRConst && offR < 0)
    return emitOpError("subview offsets must be non-negative");
  if (offCConst && offC < 0)
    return emitOpError("subview offsets must be non-negative");

  bool hasValidRow = static_cast<bool>(getValidRow());
  bool hasValidCol = static_cast<bool>(getValidCol());
  if (hasValidRow != hasValidCol)
    return emitOpError(
        "subview expects valid_row and valid_col to be both present or both absent");

  if (hasValidRow) {
    int64_t vRow = 0, vCol = 0;
    if (getConstIndex(getValidRow(), vRow)) {
      if (vRow < 0)
        return emitOpError("valid_row must be non-negative when constant");
      if (vRow > sizeR)
        return emitOpError("valid_row must be <= subview row size");
    }
    if (getConstIndex(getValidCol(), vCol)) {
      if (vCol < 0)
        return emitOpError("valid_col must be non-negative when constant");
      if (vCol > sizeC)
        return emitOpError("valid_col must be <= subview col size");
    }
  }

  auto dstShape = dstTy.getShape();
  if (dstShape.size() != 2)
    return emitOpError("expects result to be rank-2");
  auto srcShape = srcTy.getShape();
  if (srcShape.size() != 2)
    return emitOpError("expects source to be rank-2");
  if (dstShape[0] != sizeR || dstShape[1] != sizeC)
    return emitOpError("expects result shape to match subview sizes");

  if (dstTy.getElementType() != srcTy.getElementType())
    return emitOpError("expects result element type to match source");
  if (dstTy.getMemorySpace() != srcTy.getMemorySpace())
    return emitOpError("expects result address space to match source");
  auto srcCfg = srcTy.getConfigAttr();
  if (!srcCfg) srcCfg = TileBufConfigAttr::getDefault(getContext());
  auto dstCfg = dstTy.getConfigAttr();
  if (!dstCfg) dstCfg = TileBufConfigAttr::getDefault(getContext());
  if (dstCfg != srcCfg)
    return emitOpError("expects result tile config to match source");

  // Design choice: when valid[...] is omitted, infer result valid_shape from
  // subview sizes directly. We intentionally do not constrain it by source
  // valid_shape to allow user-controlled subview semantics.

  auto expectedValidDim = [&](Value explicitValid, int64_t defaultSize) {
    if (!explicitValid)
      return defaultSize;
    int64_t c = 0;
    if (getConstIndex(explicitValid, c))
      return std::min<int64_t>(c, defaultSize);
    return ShapedType::kDynamic;
  };
  int64_t expectedVRow = expectedValidDim(getValidRow(), sizeR);
  int64_t expectedVCol = expectedValidDim(getValidCol(), sizeC);
  auto dstValid = dstTy.getValidShape();
  if (dstValid.size() != 2)
    return emitOpError("expects result to have rank-2 valid_shape");
  // With the valid operand omitted, the result type is authoritative for the
  // valid extent: accept any static value in [0, size] (this subsumes both the
  // full-size default and the v=0 no-op-replay empty marker). Lowering derives
  // the bind_tile valid operand from this type. A dynamic result valid still
  // requires an explicit operand to supply the runtime extent, so it stays
  // rejected on this path.
  bool rowInferred = !getValidRow() && dstValid[0] != ShapedType::kDynamic &&
                     dstValid[0] >= 0 && dstValid[0] <= sizeR;
  bool colInferred = !getValidCol() && dstValid[1] != ShapedType::kDynamic &&
                     dstValid[1] >= 0 && dstValid[1] <= sizeC;
  if (dstValid[0] != expectedVRow && !rowInferred)
    return emitOpError("expects result valid_shape[0] to match inferred/explicit valid_row");
  if (dstValid[1] != expectedVCol && !colInferred)
    return emitOpError("expects result valid_shape[1] to match inferred/explicit valid_col");

  auto cfg = srcTy.getConfigAttr();
  if (!cfg) cfg = TileBufConfigAttr::getDefault(getContext());

  int64_t innerRows = 1, innerCols = 1;
  bool boxed = false;
  int32_t bl = 0, sl = 0;
  if (failed(computeInnerShape(cfg, srcTy.getElementType(), innerRows, innerCols,
                               boxed, bl, sl)))
    return emitOpError("unsupported tile layout for subview");

  if (!boxed)
    return success();

  // Boxed layout: require static 2D sizes with inner alignment. Offsets may be
  // dynamic, but static offsets must be aligned.
  if (sizeR % innerRows != 0 || sizeC % innerCols != 0)
    return emitOpError("boxed layout subview sizes must be multiples of inner shape");

  if (offRConst) {
    if (offR % innerRows != 0)
      return emitOpError("boxed layout subview offsets must be multiples of inner shape");
  }
  if (offCConst) {
    if (offC % innerCols != 0)
      return emitOpError("boxed layout subview offsets must be multiples of inner shape");
  }

  (void)bl;
  if (srcShape.size() != 2 ||
      srcShape[0] == ShapedType::kDynamic ||
      srcShape[1] == ShapedType::kDynamic) {
    return emitOpError("boxed layout subview requires static source shape");
  }

  return success();
}

} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;
 
// =============================================================================
// Helper Functions
// =============================================================================
 
[[maybe_unused]] static AddressSpace getAddressSpace(Value val) {
  auto type = llvm::dyn_cast<MemRefType>(val.getType());
  if (!type) return AddressSpace::Zero; // Default
 
  // 假设你的 AddressSpaceAttr 存储在 MemRef 的 memorySpace 中
  // 需要根据你的 getPTOAddressSpaceAttr 实现来调整
  auto attr = llvm::dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace());
  if (attr) return attr.getAddressSpace();
  return AddressSpace::Zero;
}
 
// =============================================================================
// Side Effects Implementation
// =============================================================================
 
// [Fix] 辅助函数：重载以支持 OpOperand* 和 OpResult，避免直接传 Value
 
// 针对操作数 (Operand) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpOperand *operand, MemoryEffects::Effect *effect) {
  if (operand)
    effects.emplace_back(effect, operand, SideEffects::DefaultResource::get());
}
 
// 针对结果 (Result) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpResult result, MemoryEffects::Effect *effect) {
  if (result)
    effects.emplace_back(effect, result, SideEffects::DefaultResource::get());
}

// === TLoadOp ===
// Read: src, Write: dst
// 针对 OpOperand* 的重载
void TLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // [Fix] 单个操作数，直接取地址
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

void TPrefetchOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TAbsOp ===
// Read: src, Write: dst
void TAbsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TStoreOp ===
// Read: src, Write: dst (GM)
void TStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto preQuantRange = getPreQuantScalarMutable();
  if (!preQuantRange.empty())
    addEffect(effects, &*preQuantRange.begin(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMovOp ===
// Read: src, Write: dst
void TMovOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto fpRange = getFpMutable();
  if (!fpRange.empty())
    addEffect(effects, &*fpRange.begin(), MemoryEffects::Read::get());
  auto preQuantRange = getPreQuantScalarMutable();
  if (!preQuantRange.empty())
    addEffect(effects, &*preQuantRange.begin(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

#define PTO_ADD_READ(operand) addEffect(effects, &(operand), MemoryEffects::Read::get())
#define PTO_ADD_WRITE(operand) addEffect(effects, &(operand), MemoryEffects::Write::get())

#define PTO_DEFINE_UNARY_EFFECTS(OpClass, srcOperand, dstOperand)                    \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(srcOperand);                                                       \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_BINARY_EFFECTS(OpClass, lhsOperand, rhsOperand, dstOperand)       \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(lhsOperand);                                                       \
    PTO_ADD_READ(rhsOperand);                                                       \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_TERNARY_EFFECTS(OpClass, op0, op1, op2, dstOperand)               \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(op0);                                                              \
    PTO_ADD_READ(op1);                                                              \
    PTO_ADD_READ(op2);                                                              \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_QUATERNARY_EFFECTS(OpClass, op0, op1, op2, op3, dstOperand)      \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(op0);                                                              \
    PTO_ADD_READ(op1);                                                              \
    PTO_ADD_READ(op2);                                                              \
    PTO_ADD_READ(op3);                                                              \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

void LoadScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getPtrMutable());
}

void StoreScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getPtrMutable());
}

// === Tile/Device ops added for InsertSync ===

// MGATHER: Read(mem, idx) -> Write(dst)
void MGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMemMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// MSCATTER: Read(src, idx) -> Write(mem)
void MScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getMemMutable());
}

// TGETVAL: Read(src) -> scalar result
void TGetValOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
}

void THistogramOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TGetScaleAddrOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TSETVAL: Write(dst) (single element update)
void TSetValOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// SET_VALIDSHAPE: update runtime valid row/col metadata on source tile in-place.
void SetValidShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getSourceMutable());
}

// GET_VALIDSHAPE: read runtime valid row/col metadata from source tile.
void GetValidShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSourceMutable());
}

// Elementwise + reductions: mostly PIPE_V tilebuf ops
PTO_DEFINE_BINARY_EFFECTS(TAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TAddCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TAddSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TAddSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
void TAxpyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getScalarMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TAndOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TConcatOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_QUATERNARY_EFFECTS(TConcatidxOp, getSrc0Mutable(), getSrc1Mutable(), getSrc0IdxMutable(), getSrc1IdxMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TAndSOp, getSrcMutable(), getDstMutable())

// TCI: Write(dst) (generates sequence)
void TCIOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// TTRI: Write(dst) (generates triangular mask)
void TTriOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TCmpOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TCmpSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_UNARY_EFFECTS(TColExpandOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandExpdifOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColMaxOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColMinOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColProdOp, getSrcMutable(), getDstMutable())

void TColArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TColArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TColSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TCvtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}
void TRandomOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}
PTO_DEFINE_BINARY_EFFECTS(TDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TDIVS has custom assembly format; conservatively treat first 2 operands as reads.
void TDivSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getScalarMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TExpOp, getSrcMutable(), getDstMutable())

// TEXPANDS: Write(dst) (broadcast scalar)
void TExpandsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// TEXTRACT: Read(src) -> Write(dst)
void TExtractOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TINSERT: Read(src) -> Write(dst)
void TInsertOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto fpRange = getFpMutable();
  if (!fpRange.empty())
    addEffect(effects, &*fpRange.begin(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// TEXTRACT_FP: Read(src), Read(fp) -> Write(dst)
void TExtractFPOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getFpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TINSERT_FP: Read(src), Read(fp) -> Write(dst)
void TInsertFPOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getFpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TFillPadOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFillPadExpandOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFillPadInplaceOp, getSrcMutable(), getDstMutable())

void TGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (auto cdst = getCdstMutable(); !cdst.empty())
    PTO_ADD_WRITE(cdst[0]);
  if (auto indices = getIndicesMutable(); !indices.empty())
    PTO_ADD_READ(indices[0]);
  if (auto tmp = getTmpMutable(); !tmp.empty())
    PTO_ADD_READ(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TGatherBOp, getSrcMutable(), getOffsetsMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLogOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLReluOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMaxSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMinSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TMovFPOp, getSrcMutable(), getFpMutable(), getDstMutable())

void TMrgSortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (auto &opnd : getSrcsMutable()) {
    PTO_ADD_READ(opnd);
  }
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  for (auto &opnd : getDstsMutable()) {
    PTO_ADD_WRITE(opnd);
  }
  auto executed = getExcutedMutable();
  if (!executed.empty()) {
    PTO_ADD_WRITE(executed[0]);
  }
}

PTO_DEFINE_BINARY_EFFECTS(TMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMulSOp, getSrc0Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TNegOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TNotOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TOrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TOrSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TPartAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TPartMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TPartMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
void TPartArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_READ(getSrc0IdxMutable());
  PTO_ADD_READ(getSrc1IdxMutable());
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getDstIdxMutable());
}
void TPartArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_READ(getSrc0IdxMutable());
  PTO_ADD_READ(getSrc1IdxMutable());
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getDstIdxMutable());
}
PTO_DEFINE_BINARY_EFFECTS(TPartMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
// TPRELU: Read(src0, src1) -> Write(tmp, dst)
void TPReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 pto-isa TPRELU implementation does not consume tmp; modeling tmp as a
  // write-only scratch on A5 incorrectly inflates local-memory planning and
  // can trigger false vec-overflow diagnostics.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TQuantOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getFpMutable());
  auto offsetRange = getOffsetMutable();
  if (!offsetRange.empty())
    PTO_ADD_READ(offsetRange[0]);
  PTO_ADD_WRITE(getDstMutable());
}
PTO_DEFINE_TERNARY_EFFECTS(TDequantOp, getSrcMutable(), getScaleMutable(),
                           getOffsetMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TRecipOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TReluOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TFModOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFModSOp, getSrcMutable(), getDstMutable())
void TRemOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRemSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}
PTO_DEFINE_UNARY_EFFECTS(TRowExpandOp, getSrcMutable(), getDstMutable())

void TRowExpandDivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandSubOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TRowExpandAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

void TRowExpandExpdifOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

// Row reductions use tmp scratch tile.
void TRowMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMAX; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMIN; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  if (getTargetArch(getOperation()) != PTOArch::A5)
    PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TRowProdOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}
void TRsqrtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

void TScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (getIndexes()) {
    auto idx = getIndexesMutable();
    if (!idx.empty())
      PTO_ADD_READ(idx[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

// Select: Read(mask, src0, src1) -> Write(tmp, dst)
void TSelOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TSELS: Read(src0, src1) -> Write(tmp, dst)
void TSelSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TShlOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TShrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShlSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShrSOp, getSrcMutable(), getDstMutable())

// TSORT32: Read(src, idx) -> Write(dst [, tmp])
void TSort32Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty())
    PTO_ADD_WRITE(tmp[0]);
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TSqrtOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TSubCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TSubSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TXORS: Read(src) -> Write(tmp, dst)
void TXorSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TXOR: Read(src0, src1) -> Write(tmp?, dst)
void TXorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TTRANS: Read(src) -> Write(tmp, dst)
void TTransOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getTmpMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TPrintOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getSrcMutable());
}

#undef PTO_DEFINE_TERNARY_EFFECTS
#undef PTO_DEFINE_BINARY_EFFECTS
#undef PTO_DEFINE_UNARY_EFFECTS
#undef PTO_ADD_WRITE
#undef PTO_ADD_READ

// === TMatmulOp ===
// Read: lhs, rhs, (bias), Write: dst
void TMatmulOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Singleton -> 直接取地址
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulAccOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TMatmulAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAccInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulBiasOp ===
// Read: a, b, bias, Write: dst
void TMatmulBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  // 这里的 bias 是必选的 AnyType:$bias，所以是 Singleton
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvOp ===
// Read: lhs, rhs, Write: dst
void TGemvOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvAccOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TGemvAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAccInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvBiasOp ===
// Read: a, b, bias, Write: dst
void TGemvBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxOp ===
// Read: a, a_scale, b, b_scale, Write: dst
void TGemvMxOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxAccOp ===
// Read: c_in, a, a_scale, b, b_scale, Write: dst
void TGemvMxAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getCInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxBiasOp ===
// Read: a, a_scale, b, b_scale, bias, Write: dst
void TGemvMxBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulOp ===
void TMatmulMxOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulAccMxOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TMatmulMxAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getCInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulBiasMxOp ===
// Read: a, b, bias, Write: dst
void TMatmulMxBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  // 这里的 bias 是必选的 AnyType:$bias，所以是 Singleton
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

static bool isInsideSectionCube(Operation *op) {
  return op->getParentOfType<pto::SectionCubeOp>() != nullptr;
}

static bool isInsideSectionVector(Operation *op) {
  return op->getParentOfType<pto::SectionVectorOp>() != nullptr;
}

static std::optional<FunctionKernelKind>
getEnclosingFunctionKernelKind(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return std::nullopt;

  auto kernelKindAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(
          FunctionKernelKindAttr::name);
  if (!kernelKindAttr)
    return std::nullopt;

  return kernelKindAttr.getKernelKind();
}

static bool isInsideSectionOrAttributedKernel(Operation *op) {
  return isInsideSectionCube(op) || isInsideSectionVector(op) ||
         getEnclosingFunctionKernelKind(op).has_value();
}

static LogicalResult verifySplitAttr(Operation *op, int64_t split) {
  if (split < 0 || split > 2)
    return op->emitOpError("expects 'split' to be 0, 1, or 2");
  return success();
}

static LogicalResult verifyFrontendKernelKind(Operation *op,
                                              FunctionKernelKind expected,
                                              StringRef kernelName) {
  auto kernelKind = getEnclosingFunctionKernelKind(op);
  if (!kernelKind || *kernelKind != expected) {
    return op->emitOpError("must be inside a ")
           << kernelName << " kernel function";
  }
  return success();
}

static ParseResult parseFrontendInitializePipeOp(OpAsmParser &parser,
                                                 OperationState &result) {
  NamedAttrList attrs;
  bool sawId = false;
  bool sawDirMask = false;
  bool sawSlotSize = false;
  bool sawSlotNum = false;
  bool sawLocalSlotNum = false;
  bool sawNoSplit = false;

  if (parser.parseLBrace())
    return failure();

  while (failed(parser.parseOptionalRBrace())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual())
      return failure();

    if (keyword == "id") {
      if (sawId)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'id' clause");
      IntegerAttr idAttr;
      if (parser.parseAttribute(idAttr, parser.getBuilder().getI32Type(), "id",
                                attrs))
        return failure();
      sawId = true;
    } else if (keyword == "dir_mask") {
      if (sawDirMask)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'dir_mask' clause");
      IntegerAttr dirMaskAttr;
      if (parser.parseAttribute(dirMaskAttr, parser.getBuilder().getI8Type(),
                                "dir_mask", attrs))
        return failure();
      sawDirMask = true;
    } else if (keyword == "slot_size") {
      if (sawSlotSize)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'slot_size' clause");
      IntegerAttr slotSizeAttr;
      if (parser.parseAttribute(slotSizeAttr, parser.getBuilder().getI32Type(),
                                "slot_size", attrs))
        return failure();
      sawSlotSize = true;
    } else if (keyword == "slot_num") {
      if (sawSlotNum)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'slot_num' clause");
      IntegerAttr slotNumAttr;
      if (parser.parseAttribute(slotNumAttr, parser.getBuilder().getI32Type(),
                                "slot_num", attrs))
        return failure();
      sawSlotNum = true;
    } else if (keyword == "local_slot_num") {
      if (sawLocalSlotNum)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'local_slot_num' clause");
      IntegerAttr localSlotNumAttr;
      if (parser.parseAttribute(localSlotNumAttr, parser.getBuilder().getI32Type(),
                                "local_slot_num", attrs))
        return failure();
      sawLocalSlotNum = true;
    } else if (keyword == "nosplit") {
      if (sawNoSplit)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'nosplit' clause");
      BoolAttr noSplitAttr;
      if (parser.parseAttribute(noSplitAttr, "nosplit", attrs))
        return failure();
      sawNoSplit = true;
    } else {
      return parser.emitError(parser.getCurrentLocation())
             << "unexpected keyword '" << keyword << "'";
    }

    if (succeeded(parser.parseOptionalRBrace()))
      break;
    if (parser.parseComma())
      return failure();
  }

  if (!sawDirMask)
    return parser.emitError(parser.getNameLoc(), "expected 'dir_mask' clause");
  if (!sawSlotSize)
    return parser.emitError(parser.getNameLoc(), "expected 'slot_size' clause");
  if (!sawId)
    attrs.set("id", parser.getBuilder().getI32IntegerAttr(0));

  OpAsmParser::UnresolvedOperand gmSlotBuffer;
  OpAsmParser::UnresolvedOperand gmSlotTensor;
  OpAsmParser::UnresolvedOperand c2vConsumerBuf;
  OpAsmParser::UnresolvedOperand v2cConsumerBuf;
  Type gmSlotBufferTy;
  Type gmSlotTensorTy;
  Type c2vConsumerBufTy;
  Type v2cConsumerBufTy;
  bool hasGmSlotBuffer = false;
  bool hasGmSlotTensor = false;
  bool hasC2vConsumerBuf = false;
  bool hasV2cConsumerBuf = false;

  if (parser.parseLParen())
    return failure();
  while (failed(parser.parseOptionalRParen())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual())
      return failure();

    if (keyword == "gm_slot_buffer") {
      if (hasGmSlotBuffer)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'gm_slot_buffer' operand");
      if (parser.parseOperand(gmSlotBuffer) ||
          parser.parseColonType(gmSlotBufferTy))
        return failure();
      hasGmSlotBuffer = true;
    } else if (keyword == "gm_slot_tensor") {
      if (hasGmSlotTensor)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'gm_slot_tensor' operand");
      if (parser.parseOperand(gmSlotTensor) ||
          parser.parseColonType(gmSlotTensorTy))
        return failure();
      hasGmSlotTensor = true;
    } else if (keyword == "c2v_consumer_buf") {
      if (hasC2vConsumerBuf)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'c2v_consumer_buf' operand");
      if (parser.parseOperand(c2vConsumerBuf) ||
          parser.parseColonType(c2vConsumerBufTy))
        return failure();
      hasC2vConsumerBuf = true;
    } else if (keyword == "v2c_consumer_buf") {
      if (hasV2cConsumerBuf)
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'v2c_consumer_buf' operand");
      if (parser.parseOperand(v2cConsumerBuf) ||
          parser.parseColonType(v2cConsumerBufTy))
        return failure();
      hasV2cConsumerBuf = true;
    } else {
      return parser.emitError(parser.getCurrentLocation())
             << "unexpected initialize_pipe operand '" << keyword << "'";
    }

    if (succeeded(parser.parseOptionalRParen()))
      break;
    if (parser.parseComma())
      return failure();
  }

  if (parser.parseOptionalAttrDict(attrs))
    return failure();

  result.addAttributes(attrs);
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {hasGmSlotBuffer ? 1 : 0, hasGmSlotTensor ? 1 : 0,
                           hasC2vConsumerBuf ? 1 : 0,
                           hasV2cConsumerBuf ? 1 : 0}));
  if (hasGmSlotBuffer &&
      parser.resolveOperand(gmSlotBuffer, gmSlotBufferTy, result.operands))
    return failure();
  if (hasGmSlotTensor &&
      parser.resolveOperand(gmSlotTensor, gmSlotTensorTy, result.operands))
    return failure();
  if (hasC2vConsumerBuf &&
      parser.resolveOperand(c2vConsumerBuf, c2vConsumerBufTy, result.operands))
    return failure();
  if (hasV2cConsumerBuf &&
      parser.resolveOperand(v2cConsumerBuf, v2cConsumerBufTy, result.operands))
    return failure();
  return success();
}

template <typename InitOpT>
static void printFrontendInitializePipeOp(InitOpT op, OpAsmPrinter &p) {
  p << " {";
  bool needsComma = false;
  auto printClause = [&](StringRef keyword, auto value) {
    if (needsComma)
      p << ", ";
    p << keyword << " = " << value;
    needsComma = true;
  };

  if (op.getId() != 0)
    printClause("id", op.getId());
  printClause("dir_mask", static_cast<int32_t>(op.getDirMask()));
  printClause("slot_size", op.getSlotSize());
  if (auto slotNumAttr = op.getSlotNumAttr())
    printClause("slot_num", slotNumAttr.getInt());
  if (auto localSlotNumAttr = op.getLocalSlotNumAttr())
    printClause("local_slot_num", localSlotNumAttr.getInt());
  if (auto noSplitAttr = op.getNosplitAttr())
    printClause("nosplit", noSplitAttr.getValue() ? "true" : "false");
  p << "}";

  p << "(";
  bool needsOperandComma = false;
  auto printOperandClause = [&](StringRef keyword, Value value) {
    if (needsOperandComma)
      p << ", ";
    p << keyword << " = " << value << " : " << value.getType();
    needsOperandComma = true;
  };
  if (op.getGmSlotBuffer()) {
    printOperandClause("gm_slot_buffer", op.getGmSlotBuffer());
  }
  if (op.getGmSlotTensor())
    printOperandClause("gm_slot_tensor", op.getGmSlotTensor());
  if (op.getC2vConsumerBuf())
    printOperandClause("c2v_consumer_buf", op.getC2vConsumerBuf());
  if (op.getV2cConsumerBuf())
    printOperandClause("v2c_consumer_buf", op.getV2cConsumerBuf());
  p << ")";
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"id", "dir_mask", "slot_size", "slot_num",
                       "local_slot_num",
                       "nosplit", "operandSegmentSizes"});
}

static std::optional<uint64_t>
getStaticElementCount(ArrayRef<int64_t> shape) {
  uint64_t count = 1;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim < 0)
      return std::nullopt;
    count *= static_cast<uint64_t>(dim);
  }
  return count;
}

static bool isSameOrHalfSlotByteSize(uint64_t tensorBytes, uint64_t slotBytes) {
  return tensorBytes == slotBytes || tensorBytes * 2 == slotBytes;
}

static LogicalResult verifyFrontendGlobalSlotTensor(Operation *op, Value tensor,
                                                    int8_t dirMask,
                                                    int32_t slotSize) {
  (void)dirMask;
  auto tvTy = dyn_cast<TensorViewType>(tensor.getType());
  if (!tvTy)
    return op->emitOpError("expects 'gm_slot_tensor' to be !pto.tensor_view");

  ArrayRef<int64_t> shape = tvTy.getShape();
  if (shape.empty())
    return op->emitOpError(
        "expects 'gm_slot_tensor' to describe one slot entry tensor");

  if (auto elemCount = getStaticElementCount(shape)) {
    uint64_t elemBytes = getElemByteSize(tvTy.getElementType());
    if (elemBytes != 0) {
      uint64_t tensorBytes = *elemCount * elemBytes;
      if (!isSameOrHalfSlotByteSize(tensorBytes,
                                    static_cast<uint64_t>(slotSize))) {
        return op->emitOpError()
               << "expects 'slot_size' to equal gm_slot_tensor byte size "
                  "or twice gm_slot_tensor byte size for split GlobalTensor "
                  "entries (got slot_size = "
               << slotSize << ", gm_slot_tensor byte size = " << tensorBytes
               << ")";
      }
    }
  }

  return success();
}

template <typename InitOpT>
static LogicalResult verifyFrontendInitCommon(InitOpT op,
                                              FunctionKernelKind expected,
                                              StringRef kernelName) {
  if (failed(verifyFrontendKernelKind(op.getOperation(), expected, kernelName)))
    return failure();

  auto funcOp = op->template getParentOfType<func::FuncOp>();
  if (!funcOp)
    return op.emitOpError("must be nested under a func.func");

  if (op.getId() < 0)
    return op.emitOpError("expects 'id' to be non-negative");

  unsigned sameIdInitCount = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == op.getId())
        ++sameIdInitCount;
      return;
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate))
      if (aiv.getId() == op.getId())
        ++sameIdInitCount;
  });
  if (sameIdInitCount > 1) {
    return op.emitOpError(
        "requires 'id' to be unique across frontend initialize_pipe ops in the function");
  }

  int8_t dirMask = op.getDirMask();
  if (dirMask != 1 && dirMask != 2 && dirMask != 3)
    return op.emitOpError("expects 'dir_mask' to be 1, 2, or 3");
  if (op.getSlotSize() <= 0)
    return op.emitOpError("expects 'slot_size' to be greater than 0");
  int32_t slotNum = dirMask == 3 ? 4 : 8;
  if (auto slotNumAttr = op.getSlotNumAttr()) {
    slotNum = slotNumAttr.getInt();
    if (slotNum <= 0)
      return op.emitOpError("expects 'slot_num' to be greater than 0");
  }
  PTOArch arch = getTargetArch(op.getOperation());

  bool hasGlobalSlotTensor = static_cast<bool>(op.getGmSlotTensor());
  bool hasC2vConsumerBuf = static_cast<bool>(op.getC2vConsumerBuf());
  bool hasV2cConsumerBuf = static_cast<bool>(op.getV2cConsumerBuf());
  if (hasGlobalSlotTensor) {
    if (op.getGmSlotBuffer() || hasC2vConsumerBuf || hasV2cConsumerBuf) {
      return op.emitOpError(
          "globaltensor pipe init expects only 'gm_slot_tensor' and no "
          "'gm_slot_buffer', 'c2v_consumer_buf', or 'v2c_consumer_buf'");
    }
    if (op.getLocalSlotNumAttr())
      return op.emitOpError(
          "globaltensor pipe init does not use 'local_slot_num'");
    if (arch == PTOArch::A5) {
      return op.emitOpError(
          "globaltensor pipe entries are supported for a2/a3 l2g2l pipes");
    }
    return verifyFrontendGlobalSlotTensor(
        op.getOperation(), op.getGmSlotTensor(), dirMask, op.getSlotSize());
  }

  if (hasC2vConsumerBuf != hasV2cConsumerBuf) {
    return op.emitOpError(
        "expects 'c2v_consumer_buf' and 'v2c_consumer_buf' to be provided together");
  }
  if (!hasC2vConsumerBuf) {
    return op.emitOpError(
        "expects local pipe init to provide 'c2v_consumer_buf' and "
        "'v2c_consumer_buf'; use 'gm_slot_tensor' for globaltensor pipe entries");
  }

  if (auto localSlotNumAttr = op.getLocalSlotNumAttr()) {
    if (arch == PTOArch::A5)
      return op.emitOpError(
          "'local_slot_num' is only supported for a2/a3 frontend pipe lowering");
    int32_t localSlotNum = localSlotNumAttr.getInt();
    if (localSlotNum <= 0)
      return op.emitOpError("expects 'local_slot_num' to be greater than 0");
    if (localSlotNum > slotNum) {
      return op.emitOpError()
             << "expects 'local_slot_num' to be less than or equal to slot_num ("
             << slotNum << ") for dir_mask = " << static_cast<int>(dirMask);
    }
  }

  return success();
}

ParseResult AicInitializePipeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseFrontendInitializePipeOp(parser, result);
}

void AicInitializePipeOp::print(OpAsmPrinter &p) {
  printFrontendInitializePipeOp(*this, p);
}

ParseResult AivInitializePipeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseFrontendInitializePipeOp(parser, result);
}

void AivInitializePipeOp::print(OpAsmPrinter &p) {
  printFrontendInitializePipeOp(*this, p);
}

static ReserveBufferOp findReserveBufferByName(func::FuncOp funcOp,
                                               StringRef name) {
  ReserveBufferOp found;
  funcOp.walk([&](ReserveBufferOp reserveOp) {
    if (reserveOp.getName() != name)
      return WalkResult::advance();
    found = reserveOp;
    return WalkResult::interrupt();
  });
  return found;
}

LogicalResult ReserveBufferOp::verify() {
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return emitOpError("must be nested under a func.func");

  if (getSize() <= 0)
    return emitOpError("expects 'size' to be greater than 0");

  auto location = getLocation().getAddressSpace();
  if (location != AddressSpace::VEC && location != AddressSpace::MAT)
    return emitOpError("expects 'location' to be #pto.address_space<vec> or #pto.address_space<mat>");

  if (!getAutoAlloc() && !getBaseAttr())
    return emitOpError("expects 'base' when 'auto' is false");

  if (auto baseAttr = getBaseAttr(); baseAttr && baseAttr.getInt() < 0)
    return emitOpError("expects 'base' to be non-negative when present");

  unsigned sameNameCount = 0;
  funcOp.walk([&](ReserveBufferOp reserveOp) {
    if (reserveOp.getName() == getName())
      ++sameNameCount;
  });
  if (sameNameCount > 1)
    return emitOpError("requires 'name' to be unique within the function");

  return success();
}

LogicalResult ImportReservedBufferOp::verify() {
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return emitOpError("must be nested under a func.func");

  auto peerFunc = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      getOperation(), getPeerFuncAttr());
  if (!peerFunc)
    return emitOpError("expects 'peer_func' to reference an existing func.func");

  unsigned sameImportCount = 0;
  funcOp.walk([&](ImportReservedBufferOp importOp) {
    if (importOp.getName() == getName() &&
        importOp.getPeerFuncAttr() == getPeerFuncAttr()) {
      ++sameImportCount;
    }
  });
  if (sameImportCount > 1) {
    return emitOpError(
        "requires (name, peer_func) to be unique within the function");
  }

  if (!findReserveBufferByName(peerFunc, getName()))
    return emitOpError("expects matching peer reserve_buffer to exist");

  return success();
}

static FailureOr<Operation *> lookupFrontendInitOpById(Operation *op,
                                                       func::FuncOp funcOp,
                                                       int32_t id) {
  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == static_cast<uint32_t>(id)) {
        matchedInit = candidate;
        ++matchedInitCount;
      }
      return WalkResult::advance();
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (aiv.getId() == static_cast<uint32_t>(id)) {
        matchedInit = candidate;
        ++matchedInitCount;
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (matchedInitCount == 0) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match a frontend initialize_pipe op in the same function";
    return failure();
  }
  if (matchedInitCount > 1) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match exactly one frontend initialize_pipe op in the same function";
    return failure();
  }
  return matchedInit;
}

static LogicalResult verifyFrontendSplitOp(Operation *op,
                                           FunctionKernelKind expected,
                                           StringRef kernelName,
                                           int32_t id,
                                           int64_t split) {
  if (failed(verifyFrontendKernelKind(op, expected, kernelName)))
    return failure();
  if (id < 0)
    return op->emitOpError("expects 'id' to be non-negative");
  return verifySplitAttr(op, split);
}

static FailureOr<int8_t> lookupFrontendInitDirMaskById(Operation *op,
                                                       func::FuncOp funcOp,
                                                       int32_t id) {
  auto initOr = lookupFrontendInitOpById(op, funcOp, id);
  if (failed(initOr))
    return failure();
  if (auto aic = dyn_cast<AicInitializePipeOp>(*initOr))
    return aic.getDirMask();
  return cast<AivInitializePipeOp>(*initOr).getDirMask();
}

static LogicalResult verifyFrontendDataOpDirection(Operation *op, int32_t id,
                                                   bool expectC2V) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return op->emitOpError("must be nested under a func.func");

  auto dirMaskOr = lookupFrontendInitDirMaskById(op, funcOp, id);
  if (failed(dirMaskOr))
    return failure();

  int8_t dirMask = *dirMaskOr;
  if (expectC2V && dirMask != 1 && dirMask != 3) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with dir_mask = 1 or 3";
  }
  if (!expectC2V && dirMask != 2 && dirMask != 3) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with dir_mask = 2 or 3";
  }
  return success();
}

static Value getFrontendInitGmSlotTensor(Operation *initOp) {
  if (auto aic = dyn_cast<AicInitializePipeOp>(initOp))
    return aic.getGmSlotTensor();
  return cast<AivInitializePipeOp>(initOp).getGmSlotTensor();
}

static LogicalResult verifyFrontendTensorEntryMatchesInit(Operation *op,
                                                          int32_t id,
                                                          Type entryTy) {
  auto entryViewTy = dyn_cast<TensorViewType>(entryTy);
  if (!entryViewTy)
    return success();

  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return op->emitOpError("must be nested under a func.func");

  auto initOr = lookupFrontendInitOpById(op, funcOp, id);
  if (failed(initOr))
    return failure();
  Value gmSlotTensor = getFrontendInitGmSlotTensor(*initOr);
  if (!gmSlotTensor) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with 'gm_slot_tensor' when the "
              "pipe entry is !pto.tensor_view";
  }

  auto slotTensorTy = dyn_cast<TensorViewType>(gmSlotTensor.getType());
  if (!slotTensorTy)
    return op->emitOpError("expects 'gm_slot_tensor' to be !pto.tensor_view");
  if (slotTensorTy.getElementType() != entryViewTy.getElementType()) {
    return op->emitOpError()
           << "expects pipe entry element type to match gm_slot_tensor element type";
  }
  if (slotTensorTy.getRank() != entryViewTy.getRank()) {
    return op->emitOpError()
           << "expects pipe entry rank to match gm_slot_tensor rank";
  }

  ArrayRef<int64_t> slotShape = slotTensorTy.getShape();
  ArrayRef<int64_t> entryShape = entryViewTy.getShape();
  for (auto [idx, entryDim] : llvm::enumerate(entryShape)) {
    int64_t slotDim = slotShape[idx];
    if (slotDim == ShapedType::kDynamic ||
        entryDim == ShapedType::kDynamic || slotDim == entryDim)
      continue;
    return op->emitOpError()
           << "expects pipe entry dimension " << idx
           << " to match gm_slot_tensor dimension " << slotDim;
  }
  return success();
}

template <typename FrontendPopOpT>
static LogicalResult verifyFrontendPopOp(FrontendPopOpT op,
                                         FunctionKernelKind expected,
                                         StringRef kernelName,
                                         bool expectC2V) {
  if (failed(verifyFrontendSplitOp(op.getOperation(), expected, kernelName,
                                   op.getId(),
                                   op.getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(op.getOperation(), op.getId(),
                                           expectC2V)))
    return failure();
  if (failed(verifyFrontendTensorEntryMatchesInit(op.getOperation(), op.getId(),
                                                  op.getTile().getType())))
    return failure();

  bool hasValidRow = static_cast<bool>(op.getValidRow());
  bool hasValidCol = static_cast<bool>(op.getValidCol());
  if (hasValidRow != hasValidCol)
    return op.emitOpError(
        "expects valid_row and valid_col operands to be provided together");
  if (!hasValidRow)
    return success();

  if (isa<TensorViewType>(op.getTile().getType()))
    return op.emitOpError(
        "does not accept valid_row/valid_col when result is !pto.tensor_view");

  auto tileTy = dyn_cast<TileBufType>(op.getTile().getType());
  if (!tileTy)
    return op.emitOpError(
        "expects tile result to be !pto.tile_buf when valid_row/valid_col operands are provided");
  if (!tileTy.hasDynamicValid())
    return op.emitOpError(
        "expects tile result to have dynamic validShape (?, ?) when valid_row/valid_col operands are provided");
  return success();
}

static LogicalResult verifyPipeShape(Operation *op, int8_t dirMask, int32_t slotSize,
                                     int32_t slotNum,
                                     std::optional<int32_t> flagBase) {
  constexpr int32_t kMaxHardwareFlagIds = 16;
  if (dirMask != 1 && dirMask != 2 && dirMask != 3)
    return op->emitOpError("expects 'dir_mask' to be 1, 2, or 3");
  if (slotSize <= 0)
    return op->emitOpError("expects 'slot_size' to be greater than 0");
  if (slotNum <= 0)
    return op->emitOpError("expects 'slot_num' to be greater than 0");
  if (flagBase && *flagBase < 0)
    return op->emitOpError("expects 'flag_base' to be non-negative when present");
  if (flagBase) {
    int32_t flagWidth = dirMask == 3 ? 4 : 2;
    if (*flagBase + flagWidth > kMaxHardwareFlagIds) {
      return op->emitOpError()
             << "requires 'flag_base' and dir_mask to fit within "
             << kMaxHardwareFlagIds << " hardware flag ids";
    }
  }

  return success();
}

static LogicalResult verifyPipeHandleProducer(Operation *op, Value pipeHandle) {
  if (!isa<pto::PipeType>(pipeHandle.getType()))
    return op->emitOpError("expects pipe operand type !pto.pipe");
  if (!pipeHandle.getDefiningOp<InitializeL2LPipeOp>() &&
      !pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>()) {
    return op->emitOpError(
        "pipe_handle must be produced by pto.initialize_l2l_pipe or "
        "pto.initialize_l2g2l_pipe");
  }
  return success();
}

static bool getTensorLikeElementAndShape(Type ty, Type &elementType,
                                         ArrayRef<int64_t> &shape) {
  if (auto tvTy = dyn_cast<TensorViewType>(ty)) {
    elementType = tvTy.getElementType();
    shape = tvTy.getShape();
    return true;
  }
  if (auto memrefTy = dyn_cast<MemRefType>(ty)) {
    elementType = memrefTy.getElementType();
    shape = memrefTy.getShape();
    return true;
  }
  return false;
}

static LogicalResult verifyTensorEntryMatchesInternalPipeInit(Operation *op,
                                                              Value pipeHandle,
                                                              Type entryTy) {
  auto entryViewTy = dyn_cast<TensorViewType>(entryTy);
  if (!entryViewTy)
    return success();

  auto initOp = pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>();
  if (!initOp) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use a pipe produced by "
              "pto.initialize_l2g2l_pipe";
  }
  if (initOp.getLocalAddr()) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use global-only "
              "pto.initialize_l2g2l_pipe without local_addr";
  }

  Type slotElementType;
  ArrayRef<int64_t> slotShape;
  if (!getTensorLikeElementAndShape(initOp.getGmAddr().getType(),
                                    slotElementType, slotShape)) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use "
              "pto.initialize_l2g2l_pipe gm_addr with tensor/memref slot type";
  }

  if (slotElementType != entryViewTy.getElementType()) {
    return op->emitOpError()
           << "expects pipe entry element type to match initialize_l2g2l_pipe "
              "gm_addr element type";
  }
  if (slotShape.size() != static_cast<size_t>(entryViewTy.getRank())) {
    return op->emitOpError()
           << "expects pipe entry rank to match initialize_l2g2l_pipe gm_addr "
              "rank";
  }

  ArrayRef<int64_t> entryShape = entryViewTy.getShape();
  for (auto [idx, entryDim] : llvm::enumerate(entryShape)) {
    int64_t slotDim = slotShape[idx];
    if (slotDim == ShapedType::kDynamic ||
        entryDim == ShapedType::kDynamic || slotDim == entryDim)
      continue;
    return op->emitOpError()
           << "expects pipe entry dimension " << idx
           << " to match initialize_l2g2l_pipe gm_addr dimension "
           << slotDim;
  }

  if (auto entryElemCount = getStaticElementCount(entryShape)) {
    uint64_t elemBytes = getElemByteSize(entryViewTy.getElementType());
    uint64_t entryBytes = *entryElemCount * elemBytes;
    if (elemBytes != 0) {
      int8_t split = 0;
      if (auto alloc = dyn_cast<TAllocOp>(op))
        split = alloc.getSplit();
      else if (auto push = dyn_cast<TPushOp>(op))
        split = push.getSplit();
      else if (auto pop = dyn_cast<TPopOp>(op))
        split = pop.getSplit();
      else if (auto free = dyn_cast<TFreeOp>(op))
        split = free.getSplit();

      uint64_t slotBytes = static_cast<uint64_t>(initOp.getSlotSize());
      bool isSplitEntry = split != 0;
      bool byteSizeMatches =
          entryBytes == slotBytes || (isSplitEntry && entryBytes * 2 == slotBytes);
      if (!byteSizeMatches) {
        return op->emitOpError()
               << "expects pipe entry byte size to match initialize_l2g2l_pipe "
                  "slot_size"
               << (isSplitEntry ? " or half slot_size for split entries" : "")
               << " (got entry byte size = " << entryBytes
               << ", slot_size = " << initOp.getSlotSize() << ")";
      }
    }
  }

  return success();
}

LogicalResult BuildAsyncSessionOp::verify() {
  Type scratchTy = getScratch().getType();
  if (!isa<pto::TileBufType, MemRefType>(scratchTy))
    return emitOpError("expects scratch to be tile_buf or memref type");

  auto scratchSpace = getPTOMemorySpaceEnum(scratchTy);
  if (!scratchSpace || *scratchSpace != pto::AddressSpace::VEC)
    return emitOpError("expects scratch to be in vec address space");

  auto scratchShape = getShapeVec(scratchTy);
  if (scratchShape.empty() || scratchShape.size() > 2)
    return emitOpError("expects scratch to be rank-1 or rank-2");
  for (int64_t dim : scratchShape) {
    if (dim == ShapedType::kDynamic)
      return emitOpError("expects scratch to have a static shape");
  }

  auto scratchBytes = getStaticByteSize(scratchTy);
  if (!scratchBytes)
    return emitOpError("expects scratch byte size to be statically known");
  if (*scratchBytes < sizeof(uint64_t))
    return emitOpError("expects scratch to provide at least 8 bytes");

  Type workspaceElemTy;
  Type workspaceTy = getWorkspace().getType();
  if (auto ptrTy = dyn_cast<pto::PtrType>(workspaceTy)) {
    workspaceElemTy = ptrTy.getElementType();
  } else if (auto memTy = dyn_cast<MemRefType>(workspaceTy)) {
    workspaceElemTy = memTy.getElementType();
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return emitOpError("expects workspace to be in GM address space");
  } else {
    return emitOpError("expects workspace to be !pto.ptr or memref type");
  }
  if (!isByteIntegerType(workspaceElemTy))
    return emitOpError("expects workspace element type to be an 8-bit integer");

  if (auto syncIdAttr = getSyncIdAttr()) {
    int64_t syncId = syncIdAttr.getInt();
    if (syncId < 0 || syncId > 7)
      return emitOpError("expects sync_id in range [0, 7]");
  }
  if (auto blockBytesAttr = getBlockBytesAttr()) {
    if (blockBytesAttr.getInt() <= 0)
      return emitOpError("expects block_bytes to be greater than 0");
  }
  if (auto commBlockOffsetAttr = getCommBlockOffsetAttr()) {
    if (commBlockOffsetAttr.getInt() < 0)
      return emitOpError("expects comm_block_offset to be non-negative");
  }
  if (auto queueNumAttr = getQueueNumAttr()) {
    if (queueNumAttr.getInt() <= 0)
      return emitOpError("expects queue_num to be greater than 0");
  }
  if (auto channelGroupIdxAttr = getChannelGroupIdxAttr()) {
    APInt value = channelGroupIdxAttr.getValue();
    if (value.isNegative())
      return emitOpError("expects channel_group_idx to be non-negative");
    if (value.ugt(UINT32_MAX))
      return emitOpError("expects channel_group_idx to fit in uint32");
  }

  return success();
}

static LogicalResult verifyAsyncTransferOp(Operation *op, Value dst, Value src) {
  Type dstElemTy = getElemTy(dst.getType());
  Type srcElemTy = getElemTy(src.getType());
  if (!dstElemTy || !srcElemTy)
    return op->emitOpError("expects src and dst to have element types");
  if (dstElemTy != srcElemTy)
    return op->emitOpError("expects src and dst to have the same element type");
  if (failed(verifyAsyncFlatContiguous1DGMViewLike(op, dst, "dst")) ||
      failed(verifyAsyncFlatContiguous1DGMViewLike(op, src, "src")))
    return failure();
  if (getShapeVec(dst.getType()) != getShapeVec(src.getType()))
    return op->emitOpError("expects src and dst to have the same static shape");
  return success();
}

LogicalResult TPutAsyncOp::verify() {
  return verifyAsyncTransferOp(getOperation(), getDst(), getSrc());
}

LogicalResult TGetAsyncOp::verify() {
  return verifyAsyncTransferOp(getOperation(), getDst(), getSrc());
}

LogicalResult TPutOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommGlobalLike(*this, getSrc(), "src")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")))
    return failure();
  if (getElemTy(getDst().getType()) != getElemTy(getSrc().getType()))
    return emitOpError("expects src and dst to have the same element type");
  if (getShapeVec(getDst().getType()) != getShapeVec(getSrc().getType()))
    return emitOpError("expects src and dst to have the same static shape");
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType()))
    return emitOpError("expects staging tile element type to match src/dst");
  return success();
}

LogicalResult TGetOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommGlobalLike(*this, getSrc(), "src")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")))
    return failure();
  if (getElemTy(getDst().getType()) != getElemTy(getSrc().getType()))
    return emitOpError("expects src and dst to have the same element type");
  if (getShapeVec(getDst().getType()) != getShapeVec(getSrc().getType()))
    return emitOpError("expects src and dst to have the same static shape");
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType()))
    return emitOpError("expects staging tile element type to match src/dst");
  return success();
}

LogicalResult TNotifyOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal")))
    return failure();
  auto valueTy = dyn_cast<IntegerType>(getValue().getType());
  if (!valueTy || valueTy.getWidth() != 32)
    return emitOpError("expects value to be i32");
  return success();
}

LogicalResult TWaitOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal")))
    return failure();
  auto cmpTy = dyn_cast<IntegerType>(getCmpValue().getType());
  if (!cmpTy || cmpTy.getWidth() != 32)
    return emitOpError("expects cmp_value to be i32");
  return success();
}

LogicalResult TTestOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal")))
    return failure();
  auto cmpTy = dyn_cast<IntegerType>(getCmpValue().getType());
  if (!cmpTy || cmpTy.getWidth() != 32)
    return emitOpError("expects cmp_value to be i32");
  return success();
}

static LogicalResult verifySyncAllGmWorkspace(Operation *op, Value workspace,
                                              StringRef name) {
  Type ty = workspace.getType();
  if (!isa<MemRefType, pto::TensorViewType, pto::PartitionTensorViewType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be a GM memref/tensor_view/partition_view";

  if (auto memTy = dyn_cast<MemRefType>(ty)) {
    if (!memTy.hasRank())
      return op->emitOpError() << "expects " << name << " to be ranked";
    if (!isGmAddressSpaceAttr(memTy.getMemorySpace()))
      return op->emitOpError() << "expects " << name
                               << " to be in GM address space";
  }

  auto elemTy = dyn_cast<IntegerType>(getElemTy(ty));
  if (!elemTy || elemTy.getWidth() != 32)
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";

  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty())
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  for (int64_t dim : shape) {
    if (dim != ShapedType::kDynamic && dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " shape to be positive";
  }
  return success();
}

static LogicalResult verifySyncAllTileWorkspace(Operation *op, Value workspace,
                                                StringRef name,
                                                pto::AddressSpace expectedSpace) {
  Type ty = workspace.getType();
  if (!isa<pto::TileBufType, MemRefType>(ty))
    return op->emitOpError() << "expects " << name
                             << " to be tile_buf or memref type";

  if (isa<pto::TileBufType>(ty) && failed(verifyTileBufCommon(op, ty, name)))
    return failure();

  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != expectedSpace)
    return op->emitOpError() << "expects " << name << " to be in "
                             << (expectedSpace == pto::AddressSpace::VEC
                                     ? "vec"
                                     : "mat")
                             << " address space";

  Type elemTy = getElemTy(ty);
  auto intTy = dyn_cast_or_null<IntegerType>(elemTy);
  if (!intTy || intTy.getWidth() != 32)
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";

  auto shape = getShapeVec(ty);
  if (shape.empty() || shape.size() > 2)
    return op->emitOpError() << "expects " << name
                             << " to be rank-1 or rank-2";
  for (int64_t dim : shape) {
    if (dim != ShapedType::kDynamic && dim <= 0)
      return op->emitOpError() << "expects " << name
                               << " shape to be positive";
  }
  return success();
}

LogicalResult SyncAllOp::verify() {
  bool hasGm = static_cast<bool>(getGmWorkspace());
  bool hasUb = static_cast<bool>(getUbWorkspace());
  bool hasL1 = static_cast<bool>(getL1Workspace());
  auto mode = getMode().getValue();
  auto coreType = getCoreType().getValue();

  if (mode == pto::SyncAllMode::Hard) {
    if (hasGm || hasUb || hasL1 || getUsedCores())
      return emitOpError(
          "expects hard syncall to have no workspace operands or used_cores");
    return success();
  }

  if (!hasGm)
    return emitOpError("expects soft syncall to provide gm_workspace");
  if (failed(verifySyncAllGmWorkspace(getOperation(), getGmWorkspace(),
                                      "gm_workspace")))
    return failure();

  if (auto used = getUsedCores()) {
    auto intTy = dyn_cast<IntegerType>(used.getType());
    if (!intTy || intTy.getWidth() != 32)
      return emitOpError("expects used_cores to be i32");
  }

  switch (coreType) {
  case pto::SyncCoreType::AIVOnly:
    if (!hasUb || hasL1)
      return emitOpError("expects soft AIV-only syncall to use gm_workspace "
                         "+ ub_workspace only");
    return verifySyncAllTileWorkspace(getOperation(), getUbWorkspace(),
                                      "ub_workspace",
                                      pto::AddressSpace::VEC);
  case pto::SyncCoreType::AICOnly:
    if (hasUb || !hasL1)
      return emitOpError("expects soft AIC-only syncall to use gm_workspace "
                         "+ l1_workspace only");
    return verifySyncAllTileWorkspace(getOperation(), getL1Workspace(),
                                      "l1_workspace",
                                      pto::AddressSpace::MAT);
  case pto::SyncCoreType::Mix:
    if (!hasUb || !hasL1)
      return emitOpError("expects soft mixed syncall to use gm_workspace + "
                         "ub_workspace + l1_workspace");
    if (failed(verifySyncAllTileWorkspace(getOperation(), getUbWorkspace(),
                                          "ub_workspace",
                                          pto::AddressSpace::VEC)))
      return failure();
    return verifySyncAllTileWorkspace(getOperation(), getL1Workspace(),
                                      "l1_workspace",
                                      pto::AddressSpace::MAT);
  }

  llvm_unreachable("unhandled SyncCoreType");
}

LogicalResult TBroadcastOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getSrc(), "src")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group")))
    return failure();
  if (getRoot() >= static_cast<uint32_t>(getGroup().size()))
    return emitOpError("expects root to index into group operands");
  if (getSrc().getType() != getGroup().front().getType())
    return emitOpError("expects src type to match group member type");
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType()))
    return emitOpError("expects staging tile element type to match src");
  return success();
}

LogicalResult CommTGatherOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group")))
    return failure();
  if (getRoot() >= static_cast<uint32_t>(getGroup().size()))
    return emitOpError("expects root to index into group operands");
  if (getElemTy(getDst().getType()) != getElemTy(getGroup().front().getType()))
    return emitOpError("expects dst element type to match group member type");
  if (getElemTy(getPing().getType()) != getElemTy(getDst().getType()))
    return emitOpError("expects staging tile element type to match dst");
  return success();
}

LogicalResult CommTScatterOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getSrc(), "src")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group")))
    return failure();
  if (getRoot() >= static_cast<uint32_t>(getGroup().size()))
    return emitOpError("expects root to index into group operands");
  if (getElemTy(getSrc().getType()) != getElemTy(getGroup().front().getType()))
    return emitOpError("expects src element type to match group member type");
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType()))
    return emitOpError("expects staging tile element type to match src");
  return success();
}

LogicalResult TReduceOp::verify() {
  if (shouldBypassDecodedMemrefVerifier(getOperation()))
    return success();
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommStagingTileLike(*this, getAcc(), "acc")) ||
      failed(verifyCommStagingTileLike(*this, getRecvPing(), "recv_ping")) ||
      failed(verifyCommPingPongSameType(*this, getRecvPing(), getRecvPong(),
                                        "recv_ping", "recv_pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group")))
    return failure();
  if (getRoot() >= static_cast<uint32_t>(getGroup().size()))
    return emitOpError("expects root to index into group operands");
  if (getElemTy(getDst().getType()) != getElemTy(getGroup().front().getType()))
    return emitOpError("expects dst element type to match group member type");
  if (getAcc().getType() != getRecvPing().getType())
    return emitOpError("expects acc and recv_ping to have identical types");
  if (getElemTy(getAcc().getType()) != getElemTy(getDst().getType()))
    return emitOpError("expects accumulator/receive tiles to match dst element type");
  return success();
}

LogicalResult AicInitializePipeOp::verify() {
  return verifyFrontendInitCommon(*this, FunctionKernelKind::Cube, "cube");
}

LogicalResult AivInitializePipeOp::verify() {
  return verifyFrontendInitCommon(*this, FunctionKernelKind::Vector, "vector");
}

LogicalResult TAllocToAivOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getEntry().getType());
}

LogicalResult TAllocToAicOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getEntry().getType());
}

LogicalResult TPushToAivOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getTile().getType());
}

LogicalResult TPushToAicOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false)))
    return failure();
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getTile().getType());
}

LogicalResult TPopFromAicOp::verify() {
  return verifyFrontendPopOp(*this, FunctionKernelKind::Vector, "vector",
                             /*expectC2V=*/true);
}

LogicalResult TPopFromAivOp::verify() {
  return verifyFrontendPopOp(*this, FunctionKernelKind::Cube, "cube",
                             /*expectC2V=*/false);
}

LogicalResult TFreeFromAicOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true)))
    return failure();
  if (getEntry())
    return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                getEntry().getType());
  return success();
}

LogicalResult TFreeFromAivOp::verify() {
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit())))
    return failure();
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false)))
    return failure();
  if (getEntry())
    return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                getEntry().getType());
  return success();
}

LogicalResult InitializeL2G2LPipeOp::verify() {
  if (failed(verifyPipeShape(getOperation(), getDirMask(), getSlotSize(),
                             getSlotNum(),
                             getFlagBaseAttr()
                                 ? std::optional<int32_t>(getFlagBaseAttr().getInt())
                                 : std::nullopt)))
    return failure();

  if (!getLocalAddr()) {
    if (getPeerLocalAddr())
      return emitOpError("'peer_local_addr' requires 'local_addr'");
    if (getLocalSlotNumAttr())
      return emitOpError(
          "'local_slot_num' is only allowed when 'local_addr' is present");
    return success();
  }

  if (auto localSlotNumAttr = getLocalSlotNumAttr()) {
    int32_t localSlotNum = localSlotNumAttr.getInt();
    if (localSlotNum <= 0)
      return emitOpError("expects 'local_slot_num' to be greater than 0");
    if (static_cast<uint32_t>(localSlotNum) > getSlotNum())
      return emitOpError(
          "expects 'local_slot_num' to be less than or equal to slot_num");
  }

  if (getDirMask() == 3 && !getPeerLocalAddr())
    return emitOpError("expects 'peer_local_addr' when dir_mask is 3");
  if (getDirMask() != 3 && getPeerLocalAddr())
    return emitOpError("'peer_local_addr' is only allowed when dir_mask is 3");
  return success();
}

LogicalResult InitializeL2LPipeOp::verify() {
  if (failed(verifyPipeShape(getOperation(), getDirMask(), getSlotSize(),
                              getSlotNum(),
                              getFlagBaseAttr()
                                  ? std::optional<int32_t>(getFlagBaseAttr().getInt())
                                  : std::nullopt)))
    return failure();

  if (getDirMask() == 3 && !getPeerLocalAddr())
    return emitOpError("expects 'peer_local_addr' when dir_mask is 3");
  if (getDirMask() != 3 && getPeerLocalAddr())
    return emitOpError("'peer_local_addr' is only allowed when dir_mask is 3");
  return success();
}

LogicalResult TPushOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation()))
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle())))
    return failure();
  if (failed(verifySplitAttr(getOperation(), getSplit())))
    return failure();
  if (failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getTile().getType())))
    return failure();
  if (!isa<TensorViewType>(getTile().getType()) &&
      getPipe() == pto::PIPE::PIPE_UNASSIGNED)
    return emitOpError("tile type must map to a supported producer pipe");
  return success();
}

LogicalResult TAllocOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation()))
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle())))
    return failure();
  if (failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType())))
    return failure();
  return verifySplitAttr(getOperation(), getSplit());
}

LogicalResult TPopOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation()))
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle())))
    return failure();
  if (failed(verifySplitAttr(getOperation(), getSplit())))
    return failure();
  if (failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getTile().getType())))
    return failure();
  if (!isa<TensorViewType>(getTile().getType()) &&
      getPipe() == pto::PIPE::PIPE_UNASSIGNED)
    return emitOpError(
        "tile type and target arch must map to a supported consumer pipe");
  return success();
}

LogicalResult TFreeOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation()))
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle())))
    return failure();
  if (getEntry() &&
      failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType())))
    return failure();
  return verifySplitAttr(getOperation(), getSplit());
}

ParseResult TFreeOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand first;
  OpAsmParser::UnresolvedOperand pipe;
  Type firstTy;
  Type pipeTy;
  bool hasEntry = false;

  if (parser.parseLParen() || parser.parseOperand(first))
    return failure();

  if (succeeded(parser.parseOptionalComma())) {
    hasEntry = true;
    if (parser.parseOperand(pipe) || parser.parseColonType(firstTy) ||
        parser.parseComma() || parser.parseType(pipeTy) || parser.parseRParen())
      return failure();
  } else {
    if (parser.parseColonType(pipeTy) || parser.parseRParen())
      return failure();
    pipe = first;
  }

  NamedAttrList attrs;
  if (parser.parseLBrace() || parser.parseKeyword("split") ||
      parser.parseEqual())
    return failure();
  IntegerAttr splitAttr;
  if (parser.parseAttribute(splitAttr, parser.getBuilder().getI8Type(),
                            "split", attrs) ||
      parser.parseRBrace() || parser.parseOptionalAttrDict(attrs))
    return failure();

  result.addAttributes(attrs);
  if (hasEntry &&
      parser.resolveOperand(first, firstTy, result.operands))
    return failure();
  if (parser.resolveOperand(pipe, pipeTy, result.operands))
    return failure();
  return success();
}

void TFreeOp::print(OpAsmPrinter &p) {
  p << "(";
  if (getEntry()) {
    p << getEntry() << ", " << getPipeHandle() << " : "
      << getEntry().getType() << ", " << getPipeHandle().getType();
  } else {
    p << getPipeHandle() << " : " << getPipeHandle().getType();
  }
  p << ") {split = " << static_cast<int32_t>(getSplit()) << "}";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"split"});
}

static func::FuncOp getParentFunc(Operation *op) {
  return op ? op->getParentOfType<func::FuncOp>() : func::FuncOp();
}

static constexpr int64_t kSimtKeepResumeSlotLimit = 123;

static Operation *getFirstNonConstantLikeOp(Block *block) {
  if (!block)
    return nullptr;
  for (Operation &op : *block) {
    if (!op.hasTrait<OpTrait::ConstantLike>())
      return &op;
  }
  return nullptr;
}

static bool isOpInRange(Operation *op, Operation *first, Operation *last) {
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    if (cur == op)
      return true;
    if (cur == last)
      return false;
  }
  return false;
}

static std::optional<unsigned> getSimtKeepResumeRegisterCount(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() <= 32)
      return 1;
    if (intType.getWidth() == 64)
      return 2;
    return std::nullopt;
  }
  if (type.isF16() || type.isBF16() || type.isF32())
    return 1;
  return std::nullopt;
}

template <typename OpT>
static Type getSimtKeepResumeValueType(OpT op);

template <>
Type getSimtKeepResumeValueType(KeepOp op) {
  return op.getPayload().getType();
}

template <>
Type getSimtKeepResumeValueType(ResumeOp op) {
  return op.getResult().getType();
}

template <typename OpT>
static LogicalResult verifySimtKeepResumeSlotRange(OpT op) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount)
    return success();
  int64_t slot = op.getSlot();
  if (slot < 0 || slot >= kSimtKeepResumeSlotLimit)
    return op.emitOpError()
           << "requires slot in range [0, "
           << (kSimtKeepResumeSlotLimit - 1) << "]";
  if (*registerCount == 2) {
    if ((slot % 2) != 0)
      return op.emitOpError()
             << "requires an even slot for 64-bit keep/resume values";
    if (slot + 1 >= kSimtKeepResumeSlotLimit)
      return op.emitOpError()
             << "requires slot in range [0, "
             << (kSimtKeepResumeSlotLimit - 2)
             << "] for 64-bit keep/resume values";
  }
  return success();
}

template <typename OpT>
static bool overlapsEarlierSimtKeepResumeSlotUse(OpT op,
                                                 SmallVectorImpl<int64_t> &used) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount)
    return false;
  int64_t slot = op.getSlot();
  for (int64_t word = slot; word < slot + *registerCount; ++word) {
    if (llvm::is_contained(used, word))
      return true;
  }
  for (int64_t word = slot; word < slot + *registerCount; ++word)
    used.push_back(word);
  return false;
}

static LogicalResult verifyUniqueResumeGroupSlots(ResumeOp current,
                                                  Operation *first) {
  SmallVector<int64_t, 4> slots;
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    auto resume = dyn_cast<ResumeOp>(cur);
    if (!resume)
      break;
    if (overlapsEarlierSimtKeepResumeSlotUse(resume, slots) &&
        resume.getOperation() == current.getOperation())
      return current.emitOpError()
             << "duplicates an earlier slot " << resume.getSlot()
             << " in the SIMT resume prologue group";
  }
  return success();
}

static LogicalResult verifyUniqueKeepGroupSlots(KeepOp current,
                                                Operation *first,
                                                Operation *last) {
  SmallVector<int64_t, 4> slots;
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    auto keep = dyn_cast<KeepOp>(cur);
    if (!keep)
      break;
    if (overlapsEarlierSimtKeepResumeSlotUse(keep, slots) &&
        keep.getOperation() == current.getOperation())
      return current.emitOpError()
             << "duplicates an earlier slot " << keep.getSlot()
             << " in the SIMT keep epilogue group";
    if (cur == last)
      break;
  }
  return success();
}

static LogicalResult verifySimtKeepResumeCommon(Operation *op, int64_t slot) {
  func::FuncOp func = getParentFunc(op);
  if (!func || !func->hasAttr(pto::kPTOSimtEntryAttrName))
    return op->emitOpError("must appear inside a function marked with '")
           << pto::kPTOSimtEntryAttrName << "'";
  if (slot < 0 || slot >= kSimtKeepResumeSlotLimit) {
    return op->emitOpError("requires slot in range [0, ")
           << (kSimtKeepResumeSlotLimit - 1) << "]";
  }
  return success();
}

static bool isSupportedSimtKeepResumeType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth() <= 64;
  return type.isF16() || type.isBF16() || type.isF32();
}

static LogicalResult verifyInsideSimtEntry(Operation *op) {
  func::FuncOp func = getParentFunc(op);
  if (!func || !func->hasAttr(pto::kPTOSimtEntryAttrName))
    return op->emitOpError("must appear inside a function marked with '")
           << pto::kPTOSimtEntryAttrName << "'";
  return success();
}

LogicalResult SyncthreadsOp::verify() {
  return verifyInsideSimtEntry(getOperation());
}

LogicalResult ThreadfenceOp::verify() {
  return verifyInsideSimtEntry(getOperation());
}

LogicalResult ThreadfenceBlockOp::verify() {
  return verifyInsideSimtEntry(getOperation());
}

void SyncthreadsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void ThreadfenceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void ThreadfenceBlockOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

LogicalResult KeepOp::verify() {
  if (failed(verifySimtKeepResumeCommon(getOperation(), getSlot())))
    return failure();
  if (!isSupportedSimtKeepResumeType(getPayload().getType()))
    return emitOpError()
           << "supports integer scalar payloads up to 64 bits and "
              "f16/bf16/f32 payloads";
  if (failed(verifySimtKeepResumeSlotRange(*this)))
    return failure();

  Block *block = getOperation()->getBlock();
  Operation *terminator = block ? block->getTerminator() : nullptr;
  if (!terminator || !isa<func::ReturnOp>(terminator))
    return emitOpError(
        "must be placed in the SIMT epilogue before func.return");

  Operation *cur = terminator->getPrevNode();
  while (cur && isa<SyncthreadsOp>(cur))
    cur = cur->getPrevNode();
  Operation *lastKeep = cur;
  if (!lastKeep || !isa<KeepOp>(lastKeep))
    return emitOpError()
           << "must be placed in the SIMT epilogue before func.return; only "
              "'pto.syncthreads' may appear between the final 'pto.keep' group "
              "and func.return";

  Operation *firstKeep = lastKeep;
  while (Operation *prev = firstKeep->getPrevNode()) {
    if (!isa<KeepOp>(prev))
      break;
    firstKeep = prev;
  }
  if (!isOpInRange(getOperation(), firstKeep, lastKeep))
    return emitOpError()
           << "must be in the contiguous SIMT keep epilogue group immediately "
              "before optional 'pto.syncthreads' and func.return";
  if (failed(verifyUniqueKeepGroupSlots(*this, firstKeep, lastKeep)))
    return failure();
  return success();
}

LogicalResult ResumeOp::verify() {
  if (failed(verifySimtKeepResumeCommon(getOperation(), getSlot())))
    return failure();
  if (!isSupportedSimtKeepResumeType(getResult().getType()))
    return emitOpError()
           << "supports integer scalar results up to 64 bits and "
              "f16/bf16/f32 results";
  if (failed(verifySimtKeepResumeSlotRange(*this)))
    return failure();
  Block *block = getOperation()->getBlock();
  Operation *first = getFirstNonConstantLikeOp(block);
  if (!first || !isa<ResumeOp>(first))
    return emitOpError()
           << "must be in the contiguous SIMT resume prologue group after "
              "constant-like operations";

  bool found = false;
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    if (!isa<ResumeOp>(cur))
      break;
    if (cur == getOperation()) {
      found = true;
      break;
    }
  }
  if (!found)
    return emitOpError()
           << "must be in the contiguous SIMT resume prologue group after "
              "constant-like operations";
  if (failed(verifyUniqueResumeGroupSlots(*this, first)))
    return failure();
  return success();
}

void BuildAsyncSessionOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getScratchMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getWorkspaceMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPutAsyncOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TGetAsyncOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPutOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
}

void TGetOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
}

void TNotifyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getValueMutable(), MemoryEffects::Read::get());
}

void TWaitOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getCmpValueMutable(), MemoryEffects::Read::get());
}

void TTestOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getCmpValueMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TBroadcastOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
  for (OpOperand &operand : getGroupMutable())
    addEffect(effects, &operand, MemoryEffects::Write::get());
}

void CommTGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
  for (OpOperand &operand : getGroupMutable())
    addEffect(effects, &operand, MemoryEffects::Read::get());
}

void CommTScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
  for (OpOperand &operand : getGroupMutable())
    addEffect(effects, &operand, MemoryEffects::Write::get());
}

void TReduceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getAccMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAccMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getRecvPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRecvPingMutable(), MemoryEffects::Write::get());
  if (getRecvPong()) {
    auto recvPongRange = getRecvPongMutable();
    if (auto it = recvPongRange.begin(); it != recvPongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
  for (OpOperand &operand : getGroupMutable())
    addEffect(effects, &operand, MemoryEffects::Read::get());
}

void WaitAsyncEventOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEventMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TestAsyncEventOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEventMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void InitializeL2G2LPipeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getGmAddrMutable(), MemoryEffects::Read::get());
  auto localAddr = getLocalAddrMutable();
  if (!localAddr.empty())
    addEffect(effects, &*localAddr.begin(), MemoryEffects::Read::get());
  auto peerLocalAddr = getPeerLocalAddrMutable();
  if (!peerLocalAddr.empty())
    addEffect(effects, &*peerLocalAddr.begin(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void InitializeL2LPipeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getLocalAddrMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPushOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getTileMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
}

void TAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEntryMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
}

void TPopOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getTileMutable(), MemoryEffects::Write::get());
}

void TFreeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto entry = getEntryMutable();
  if (!entry.empty())
    addEffect(effects, &*entry.begin(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
}

static constexpr const char kConvertRoundingKeywords[] = "r/a/f/c/z/o/h";

static ParseResult parseConvertRounding(OpAsmParser &parser,
                                        RoundingAttr &roundingAttr) {
  StringRef roundingKeyword;
  if (parser.parseKeyword("round") || parser.parseLParen() ||
      parser.parseKeyword(&roundingKeyword) || parser.parseRParen())
    return failure();
  std::optional<Rounding> rounding = symbolizeRounding(roundingKeyword);
  if (!rounding)
    return parser.emitError(parser.getCurrentLocation())
           << "expected convert rounding to be one of "
           << kConvertRoundingKeywords;
  roundingAttr = RoundingAttr::get(parser.getContext(), *rounding);
  return success();
}

static void printConvertRounding(OpAsmPrinter &printer, Operation *op,
                                 RoundingAttr rounding) {
  printer << "round(" << stringifyRounding(rounding.getValue()) << ")";
}

static ParseResult parseConvertSaturation(OpAsmParser &parser,
                                          SaturationAttr &saturationAttr) {
  StringRef saturationKeyword;
  if (parser.parseKeyword(&saturationKeyword))
    return failure();
  std::optional<Saturation> saturation =
      symbolizeSaturation(saturationKeyword);
  if (!saturation)
    return parser.emitError(parser.getCurrentLocation())
           << "expected convert saturation to be sat or nosat";
  saturationAttr = SaturationAttr::get(parser.getContext(), *saturation);
  return success();
}

static void printConvertSaturation(OpAsmPrinter &printer, Operation *op,
                                   SaturationAttr saturation) {
  printer << stringifySaturation(saturation.getValue());
}

static ParseResult parseSignedness(OpAsmParser &parser,
                                   SignednessAttr &signedness) {
  StringRef signednessKeyword;
  if (parser.parseKeyword(&signednessKeyword))
    return failure();
  std::optional<Signedness> parsed = symbolizeSignedness(signednessKeyword);
  if (!parsed)
    return parser.emitError(parser.getCurrentLocation())
           << "expected signedness to be signed or unsigned";
  signedness = SignednessAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printSignedness(OpAsmPrinter &printer, Operation *op,
                            SignednessAttr signedness) {
  printer << stringifySignedness(signedness.getValue());
}

static OptionalParseResult parseOptionalSignedness(OpAsmParser &parser,
                                                   SignednessAttr &signedness) {
  if (succeeded(parser.parseOptionalKeyword("signed"))) {
    signedness = SignednessAttr::get(parser.getContext(), Signedness::Signed);
    return success();
  }
  if (succeeded(parser.parseOptionalKeyword("unsigned"))) {
    signedness =
        SignednessAttr::get(parser.getContext(), Signedness::Unsigned);
    return success();
  }
  return std::nullopt;
}

static void printOptionalSignedness(OpAsmPrinter &printer, Operation *op,
                                    SignednessAttr signedness) {
  printer << stringifySignedness(signedness.getValue());
}

static constexpr const char kLdL2CacheKeywords[] =
    "nmfv/nmlv/nmprs/nmpref/nakeep/naclean/nadrop/idsfv/idslv/idsprs/"
    "idspref/exfv/exlv/exprs/expref";

static constexpr const char kStL2CacheKeywords[] =
    "nmfv/nmlv/nmprs/nmred/naci/napw/napi/nared/wbhfv/wbhlv/wbhprs/"
    "wbhred/wtsfv/wtslv/wtsprs/wtsred";

static ParseResult parseL1Cache(OpAsmParser &parser, L1CacheAttr &l1cache) {
  if (failed(parser.parseOptionalKeyword("l1cache"))) {
    l1cache = L1CacheAttr::get(parser.getContext(), L1Cache::Cache);
    return success();
  }

  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen())
    return failure();
  std::optional<L1Cache> parsed = symbolizeL1Cache(keyword);
  if (!parsed)
    return parser.emitError(parser.getCurrentLocation())
           << "expected memory l1cache to be cache or uncache";
  l1cache = L1CacheAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printL1Cache(OpAsmPrinter &printer, Operation *op,
                         L1CacheAttr l1cache) {
  if (!l1cache)
    return;
  printer << "l1cache(" << stringifyL1Cache(l1cache.getValue()) << ")";
}

static ParseResult parseLdL2Cache(OpAsmParser &parser,
                                  LdL2CacheAttr &l2cache) {
  if (failed(parser.parseOptionalKeyword("l2cache"))) {
    l2cache = LdL2CacheAttr::get(parser.getContext(), LdL2Cache::NMFV);
    return success();
  }

  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen())
    return failure();
  std::optional<LdL2Cache> parsed = symbolizeLdL2Cache(keyword);
  if (!parsed)
    return parser.emitError(parser.getCurrentLocation())
           << "expected load L2 cache control to be one of "
           << kLdL2CacheKeywords;
  l2cache = LdL2CacheAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printLdL2Cache(OpAsmPrinter &printer, Operation *op,
                           LdL2CacheAttr l2cache) {
  if (!l2cache)
    return;
  printer << "l2cache(" << stringifyLdL2Cache(l2cache.getValue()) << ")";
}

static ParseResult parseStL2Cache(OpAsmParser &parser,
                                  StL2CacheAttr &l2cache) {
  if (failed(parser.parseOptionalKeyword("l2cache"))) {
    l2cache = StL2CacheAttr::get(parser.getContext(), StL2Cache::NMFV);
    return success();
  }

  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen())
    return failure();
  std::optional<StL2Cache> parsed = symbolizeStL2Cache(keyword);
  if (!parsed)
    return parser.emitError(parser.getCurrentLocation())
           << "expected store L2 cache control to be one of "
           << kStL2CacheKeywords;
  l2cache = StL2CacheAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printStL2Cache(OpAsmPrinter &printer, Operation *op,
                           StL2CacheAttr l2cache) {
  if (!l2cache)
    return;
  printer << "l2cache(" << stringifyStL2Cache(l2cache.getValue()) << ")";
}

// [Include 必须放在最后]
#include "PTO/IR/PTOInterfaces.cpp.inc"
#include "PTO/IR/VPTOInterfaces.cpp.inc"
#define GET_OP_CLASSES
#include "PTO/IR/PTOOps.cpp.inc"

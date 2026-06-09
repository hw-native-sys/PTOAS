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
#include "PTO/IR/PTODialect.h"
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

namespace pto = mlir::pto;
namespace arith = mlir::arith;

using mlir::AsmParser;
using mlir::AsmPrinter;
using mlir::Attribute;
using mlir::DataLayout;
using mlir::DataLayoutEntryListRef;
using mlir::BlockArgument;
using mlir::DictionaryAttr;
using mlir::FailureOr;
using mlir::FloatType;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::MemRefType;
using mlir::ModuleOp;
using mlir::NamedAttrList;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::Operation;
using mlir::OperationState;
using mlir::OptionalParseResult;
using mlir::ParseResult;
using mlir::StringAttr;
using mlir::ShapedType;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;
using pto::AddressSpace;
using pto::AddressSpaceAttr;
using pto::BLayoutAttr;
using pto::getPTOParserTargetArch;
using pto::getPTOStorageElemByteSize;
using pto::kPTOTargetArchAttrName;
using pto::Layout;
using pto::MaskPatternAttr;
using pto::PipeAttr;
using pto::PTOArch;
using pto::PTOParserTargetArch;
using pto::PtrType;
using pto::symbolizePIPE;
using pto::TensorViewType;
using pto::PartitionTensorViewType;
using pto::TileBufConfigAttr;
using pto::TileBufType;
using pto::TileType;
using llvm::ArrayRef;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringRef;
using llvm::TypeSwitch;
using llvm::cast;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::failure;
using llvm::failed;
using llvm::isa;
using llvm::succeeded;
using llvm::success;

namespace {
constexpr unsigned kSmallVectorInlineCapacity0 = 0;
constexpr unsigned kSmallVectorInlineCapacity1 = 1;
constexpr unsigned kSmallVectorInlineCapacity2 = 2;
constexpr unsigned kSmallVectorInlineCapacity3 = 3;
constexpr unsigned kSmallVectorInlineCapacity4 = 4;
constexpr unsigned kSmallVectorInlineCapacity5 = 5;
constexpr unsigned kSmallVectorInlineCapacity8 = 8;
constexpr unsigned kSmallVectorInlineCapacity16 = 16;
constexpr unsigned kSmallVectorInlineCapacity32 = 32;
constexpr unsigned kPTORowColRank = 2;
constexpr size_t kTGatherTmpOperandCount = 2;
constexpr size_t kTGatherMaxExtraInsOperands = 3;
constexpr size_t kCommFixedOperandCount = 2;
constexpr int64_t kPTOMatmulDimMin = 1;
constexpr int64_t kPTOMatmulDimMax = 4095;
constexpr unsigned kPTOColumnDim = 1;
constexpr int64_t kPTOMinGatherDstColumns = 256;
constexpr int64_t kPTOFloat4PackedExpansion = 2;
constexpr size_t kNumber2 = 2;
constexpr size_t kNumber3 = 3;
constexpr size_t kNumber4 = 4;
constexpr size_t kNumber5 = 5;
constexpr int64_t kNumber32 = 32;
constexpr int64_t kNumber64 = 64;

template <typename T>
using SmallVec0 = SmallVector<T, kSmallVectorInlineCapacity0>;
template <typename T>
using SmallVec1 = SmallVector<T, kSmallVectorInlineCapacity1>;
template <typename T>
using SmallVec2 = SmallVector<T, kSmallVectorInlineCapacity2>;
template <typename T>
using SmallVec3 = SmallVector<T, kSmallVectorInlineCapacity3>;
template <typename T>
using SmallVec4 = SmallVector<T, kSmallVectorInlineCapacity4>;
template <typename T>
using SmallVec5 = SmallVector<T, kSmallVectorInlineCapacity5>;
template <typename T>
using SmallVec8 = SmallVector<T, kSmallVectorInlineCapacity8>;
template <typename T>
using SmallVec16 = SmallVector<T, kSmallVectorInlineCapacity16>;
template <typename T>
using SmallVec32 = SmallVector<T, kSmallVectorInlineCapacity32>;
} // namespace

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
static SmallVec4<int64_t> getShapeVec(Type ty);
static SmallVec4<int64_t> getValidShapeVec(Type ty);
static SmallVec4<int64_t> getValidShapeVec(Value value);
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

#define GET_ENUM_CLASSES
#include "PTO/IR/PTOEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "PTO/IR/PTOTypeDefs.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "PTO/IR/PTOAttrs.cpp.inc"

#include "PTO/IR/PTODialect.cpp.inc"

void mlir::pto::PTODialect::initialize() {
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
  return llvm::TypeSize::getFixed(mlir::pto::kPTOByteBitWidth);
}

llvm::TypeSize mlir::pto::HiF8Type::getTypeSizeInBits(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  (void)dataLayout;
  (void)params;
  return getOneByteTypeSize();
}

uint64_t mlir::pto::HiF8Type::getABIAlignment(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  (void)dataLayout;
  (void)params;
  return kPTOByteSize;
}

uint64_t mlir::pto::HiF8Type::getPreferredAlignment(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  (void)dataLayout;
  (void)params;
  return kPTOByteSize;
}

llvm::TypeSize mlir::pto::F4E1M2x2Type::getTypeSizeInBits(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  (void)dataLayout;
  (void)params;
  return getOneByteTypeSize();
}

uint64_t mlir::pto::F4E1M2x2Type::getABIAlignment(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  (void)dataLayout;
  (void)params;
  return kPTOByteSize;
}

uint64_t mlir::pto::F4E1M2x2Type::getPreferredAlignment(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  (void)dataLayout;
  (void)params;
  return kPTOByteSize;
}

llvm::TypeSize mlir::pto::F4E2M1x2Type::getTypeSizeInBits(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  (void)dataLayout;
  (void)params;
  return getOneByteTypeSize();
}

uint64_t mlir::pto::F4E2M1x2Type::getABIAlignment(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  (void)dataLayout;
  (void)params;
  return kPTOByteSize;
}

uint64_t mlir::pto::F4E2M1x2Type::getPreferredAlignment(
    const DataLayout &dataLayout, DataLayoutEntryListRef params) const {
  (void)dataLayout;
  (void)params;
  return kPTOByteSize;
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

static SmallVec4<int64_t> canonicalizeTileBufValidShape(ArrayRef<int64_t> validShape) {
  SmallVec4<int64_t> canonical;
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

static LogicalResult parseShapeElemTypeForPTOType(
    OpAsmParser &parser, SmallVectorImpl<int64_t> &shape, Type &elem) {
  if (failed(parser.parseLess()))
    return failure();
  if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/true)))
    return failure();
  if (failed(parser.parseType(elem)))
    return failure();
  return parser.parseGreater();
}

static Type parseShapedPTOTypeAllowNoBang(OpAsmParser &parser, StringRef head) {
  SmallVec4<int64_t> shape;
  Type elem;
  if (failed(parseShapeElemTypeForPTOType(parser, shape, elem)))
    return Type();
  MLIRContext *ctx = parser.getContext();
  if (head == "pto.tile_view")
    return PartitionTensorViewType::get(ctx, shape, elem);
  if (head == "pto.tile")
    return TileType::get(ctx, shape, elem);
  if (head == "pto.tensor_view")
    return TensorViewType::get(ctx, shape, elem);
  return Type();
}

static Type parsePtrPTOTypeAllowNoBang(OpAsmParser &parser) {
  if (failed(parser.parseLess()))
    return Type();
  Type elem;
  if (failed(parser.parseType(elem)))
    return Type();
  if (succeeded(parser.parseOptionalComma())) {
    Attribute memorySpace;
    (void)parser.parseAttribute(memorySpace);
    parser.emitError(parser.getCurrentLocation(),
                     "!pto.ptr no longer accepts address space; use !pto.ptr<elem>");
    return Type();
  }
  if (failed(parser.parseGreater()))
    return Type();
  return PtrType::get(parser.getContext(), elem);
}

static Type parseKnownPTOTypeAllowNoBang(OpAsmParser &parser, StringRef head) {
  if (head == "pto.ptr")
    return parsePtrPTOTypeAllowNoBang(parser);
  if (head == "pto.tile_view" || head == "pto.tile" ||
      head == "pto.tensor_view") {
    return parseShapedPTOTypeAllowNoBang(parser, head);
  }
  return Type();
}

[[maybe_unused]] static mlir::Type parsePTOTypeAllowNoBang(mlir::OpAsmParser &parser) {
  Type ty;
  OptionalParseResult opt = parser.parseOptionalType(ty);
  if (opt.has_value())
    return failed(*opt) ? Type() : ty;

  StringRef head;
  if (failed(parser.parseKeyword(&head)))
    return Type();
  return parseKnownPTOTypeAllowNoBang(parser, head);
}

mlir::Type TensorViewType::parse(::mlir::AsmParser &odsParser) {
  SmallVec4<int64_t> shape;
  Type elementType;
  if (failed(
          parseShapeAndElem(odsParser, shape, elementType, /*allowDynamic=*/true)))
    return Type();
  return TensorViewType::get(odsParser.getContext(), shape, elementType);
}

void TensorViewType::print(::mlir::AsmPrinter &odsPrinter) const {
  printShapeAndElem(odsPrinter, getShape(), getElementType());
}

//===----------------------------------------------------------------------===//
// pto.tdivs custom asm supports both forms below
//   pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
//   pto.tdivs ins(%scalar, %src : f32, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
// The operand order in the op follows textual input order.
//===----------------------------------------------------------------------===//

static ParseResult parseTDivSClauses(OpAsmParser &parser,
                                     OpAsmParser::UnresolvedOperand &op0,
                                     OpAsmParser::UnresolvedOperand &op1,
                                     OpAsmParser::UnresolvedOperand &dst,
                                     Type &ty0, Type &ty1, Type &dstTy) {
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(op0) || parser.parseComma() ||
      parser.parseOperand(op1) || parser.parseColonType(ty0) ||
      parser.parseComma() || parser.parseType(ty1) || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen()) {
    return failure();
  }
  return success();
}

static ParseResult validateTDivSTypes(OpAsmParser &parser, Type ty0, Type ty1,
                                      Type dstTy) {
  auto tile0 = dyn_cast<TileBufType>(ty0);
  auto tile1 = dyn_cast<TileBufType>(ty1);
  if ((tile0 && tile1) || (!tile0 && !tile1)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected exactly one tile_buf operand and one scalar operand");
  }
  if (!dyn_cast<TileBufType>(dstTy)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected outs type to be !pto.tile_buf<...>");
  }
  return success();
}

static ParseResult resolveTDivSOperands(OpAsmParser &parser,
                                        OperationState &result,
                                        OpAsmParser::UnresolvedOperand &op0,
                                        OpAsmParser::UnresolvedOperand &op1,
                                        OpAsmParser::UnresolvedOperand &dst,
                                        Type ty0, Type ty1, Type dstTy) {
  if (parser.resolveOperand(op0, ty0, result.operands) ||
      parser.resolveOperand(op1, ty1, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  return success();
}

ParseResult mlir::pto::TDivSOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand op0, op1, dst;
  Type ty0, ty1, dstTy;
  if (parseTDivSClauses(parser, op0, op1, dst, ty0, ty1, dstTy))
    return failure();

  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs))
    return failure();
  if (validateTDivSTypes(parser, ty0, ty1, dstTy) ||
      resolveTDivSOperands(parser, result, op0, op1, dst, ty0, ty1, dstTy)) {
    return failure();
  }
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
// pto.tgather custom asm supports the three PTO-ISA forms below
//   1) index+tmp   : ins(%src, %indices, %tmp : srcTy, indicesTy, tmpTy) outs(%dst : dstTy)
//   2) compare+tmp : ins(%src, %kValue, %tmp : srcTy, scalarTy, tmpTy)
//                    outs(%dst, %cdst : dstTy, cdstTy) {cmpMode = #pto.cmp<gt>, offset = 7}
//   3) mask        : ins(%src, {maskPattern = #pto.mask_pattern<P0101>} : srcTy) outs(%dst : dstTy)
//===----------------------------------------------------------------------===//

namespace {

struct TGatherParseState {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand dst;
  OpAsmParser::UnresolvedOperand cdst;
  SmallVec3<OpAsmParser::UnresolvedOperand> insOps;
  SmallVec3<Type> insTypes;
  Type srcTy;
  Type dstTy;
  Type cdstTy;
  bool hasCdst = false;
  bool hasMask = false;
  bool hasIndices = false;
  bool hasTmp = false;
  bool hasKValue = false;
};

template <typename ParseStateT>
static ParseResult parseMaskPatternInsClause(OpAsmParser &parser,
                                             OperationState &result,
                                             ParseStateT &state) {
  if (parser.parseKeyword("maskPattern") || parser.parseEqual())
    return failure();
  Attribute rawMaskAttr;
  if (parser.parseAttribute(rawMaskAttr) || parser.parseRBrace())
    return failure();
  auto mp = llvm::dyn_cast<MaskPatternAttr>(rawMaskAttr);
  if (!mp) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected #pto.mask_pattern<Pxxxx> for maskPattern");
  }
  result.addAttribute("maskPattern", mp);
  state.hasMask = true;
  if (parser.parseColonType(state.srcTy) || parser.parseRParen())
    return failure();
  return success();
}

static ParseResult parseTGatherMaskInsClause(OpAsmParser &parser,
                                             OperationState &result,
                                             TGatherParseState &state) {
  return parseMaskPatternInsClause(parser, result, state);
}

static ParseResult parseTGatherExtraInsClause(OpAsmParser &parser,
                                              TGatherParseState &state) {
  OpAsmParser::UnresolvedOperand extra;
  if (parser.parseOperand(extra))
    return failure();
  state.insOps.push_back(extra);
  while (succeeded(parser.parseOptionalComma())) {
    if (state.insOps.size() == kTGatherMaxExtraInsOperands) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected at most 3 extra operands in tgather ins(...)");
    }
    if (parser.parseOperand(extra))
      return failure();
    state.insOps.push_back(extra);
  }
  if (parser.parseColon() || parser.parseType(state.srcTy))
    return failure();
  for (size_t i = 0; i < state.insOps.size(); ++i) {
    Type ty;
    if (parser.parseComma() || parser.parseType(ty))
      return failure();
    state.insTypes.push_back(ty);
  }
  return parser.parseRParen();
}

static ParseResult parseTGatherInsClause(OpAsmParser &parser,
                                         OperationState &result,
                                         TGatherParseState &state) {
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(state.src)) {
    return failure();
  }
  if (!succeeded(parser.parseOptionalComma())) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected ',' after src operand in ins(...)");
  }
  if (succeeded(parser.parseOptionalLBrace()))
    return parseTGatherMaskInsClause(parser, result, state);
  return parseTGatherExtraInsClause(parser, state);
}

static ParseResult parseTGatherOutsClause(OpAsmParser &parser,
                                          TGatherParseState &state) {
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(state.dst))
    return failure();
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(state.cdst))
      return failure();
    state.hasCdst = true;
  }
  if (parser.parseColonType(state.dstTy))
    return failure();
  if (state.hasCdst && (parser.parseComma() || parser.parseType(state.cdstTy)))
    return failure();
  return parser.parseRParen();
}

static ParseResult parseOptionalTGatherMaskPattern(OpAsmParser &parser,
                                                   OperationState &result,
                                                   TGatherParseState &state) {
  if (!succeeded(parser.parseOptionalKeyword("maskPattern")))
    return success();
  if (state.hasMask) {
    return parser.emitError(parser.getCurrentLocation(),
                            "maskPattern may only be specified once");
  }
  if (parser.parseEqual())
    return failure();
  Attribute rawMaskAttr;
  if (parser.parseAttribute(rawMaskAttr))
    return failure();
  auto mp = llvm::dyn_cast<MaskPatternAttr>(rawMaskAttr);
  if (!mp) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected #pto.mask_pattern<Pxxxx> for maskPattern");
  }
  result.addAttribute("maskPattern", mp);
  state.hasMask = true;
  return success();
}

static ParseResult validateTGatherMaskForm(OpAsmParser &parser,
                                           const TGatherParseState &state) {
  if (!state.insOps.empty()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "mask-pattern tgather does not take extra ins operands");
  }
  if (state.hasCdst) {
    return parser.emitError(parser.getCurrentLocation(),
                            "mask-pattern tgather expects a single outs operand");
  }
  return success();
}

static ParseResult validateTGatherCompareForm(OpAsmParser &parser,
                                              TGatherParseState &state) {
  if (state.insOps.empty() ||
      !(isa<IntegerType>(state.insTypes.front()) ||
        isa<FloatType>(state.insTypes.front()))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "compare-form tgather expects a scalar kValue operand");
  }
  state.hasKValue = true;
  if (state.insOps.size() >= kTGatherTmpOperandCount) {
    if (!isTileLikeType(state.insTypes[1])) {
      return parser.emitError(parser.getCurrentLocation(),
                              "compare-form tgather tmp must be tile-like");
    }
    state.hasTmp = true;
  }
  if (state.insOps.size() == kTGatherMaxExtraInsOperands) {
    return parser.emitError(parser.getCurrentLocation(),
                            "compare-form tgather expects at most src, kValue, tmp in ins(...)");
  }
  return success();
}

static ParseResult validateTGatherIndexForm(OpAsmParser &parser,
                                            TGatherParseState &state) {
  if (!state.insOps.empty() && !isTileLikeType(state.insTypes.front())) {
    return parser.emitError(parser.getCurrentLocation(),
                            "index-form tgather expects tile-like indices; "
                            "compare-form must use outs(dst, cdst)");
  }
  if (state.insOps.empty())
    return success();
  state.hasIndices = true;
  if (state.insOps.size() >= kTGatherTmpOperandCount) {
    if (!isTileLikeType(state.insTypes[1])) {
      return parser.emitError(parser.getCurrentLocation(),
                              "index-form tgather tmp must be tile-like");
    }
    state.hasTmp = true;
  }
  if (state.insOps.size() == kTGatherMaxExtraInsOperands) {
    return parser.emitError(parser.getCurrentLocation(),
                            "index-form tgather expects at most src, indices, tmp in ins(...)");
  }
  return success();
}

static ParseResult validateTGatherOperands(OpAsmParser &parser,
                                           TGatherParseState &state) {
  if (state.hasMask)
    return validateTGatherMaskForm(parser, state);
  if (state.hasCdst)
    return validateTGatherCompareForm(parser, state);
  return validateTGatherIndexForm(parser, state);
}

static ParseResult resolveTGatherOperands(OpAsmParser &parser,
                                          OperationState &result,
                                          const TGatherParseState &state) {
  if (parser.resolveOperand(state.src, state.srcTy, result.operands) ||
      parser.resolveOperand(state.dst, state.dstTy, result.operands))
    return failure();
  if (state.hasCdst &&
      parser.resolveOperand(state.cdst, state.cdstTy, result.operands)) {
    return failure();
  }
  if ((state.hasIndices || state.hasKValue) &&
      parser.resolveOperand(state.insOps[0], state.insTypes[0], result.operands)) {
    return failure();
  }
  if (state.hasTmp &&
      parser.resolveOperand(state.insOps[1], state.insTypes[1], result.operands)) {
    return failure();
  }
  return success();
}

static void addTGatherSegmentSizes(OpAsmParser &parser, OperationState &result,
                                   const TGatherParseState &state) {
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, 1, state.hasCdst ? 1 : 0,
                           state.hasIndices ? 1 : 0, state.hasTmp ? 1 : 0,
                           state.hasKValue ? 1 : 0}));
}

struct TScatterParseState {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand indexes;
  OpAsmParser::UnresolvedOperand dst;
  Type srcTy;
  Type idxTy;
  Type dstTy;
  bool hasMask = false;
  bool hasIndexes = false;
};

static ParseResult parseTScatterMaskInsClause(OpAsmParser &parser,
                                              OperationState &result,
                                              TScatterParseState &state) {
  return parseMaskPatternInsClause(parser, result, state);
}

static ParseResult parseTScatterIndexesInsClause(OpAsmParser &parser,
                                                 TScatterParseState &state) {
  if (parser.parseOperand(state.indexes))
    return failure();
  state.hasIndexes = true;
  if (parser.parseColon() || parser.parseType(state.srcTy) || parser.parseComma() ||
      parser.parseType(state.idxTy)) {
    return failure();
  }
  return parser.parseRParen();
}

static ParseResult parseTScatterInsClause(OpAsmParser &parser,
                                          OperationState &result,
                                          TScatterParseState &state) {
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(state.src)) {
    return failure();
  }
  if (!succeeded(parser.parseOptionalComma())) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected ',' after src operand in ins(...)");
  }
  if (succeeded(parser.parseOptionalLBrace()))
    return parseTScatterMaskInsClause(parser, result, state);
  return parseTScatterIndexesInsClause(parser, state);
}

static ParseResult validateTScatterOperands(OpAsmParser &parser,
                                            const TScatterParseState &state) {
  if (state.hasMask && state.hasIndexes) {
    return parser.emitError(parser.getCurrentLocation(),
                            "mask-pattern tscatter does not take indexes");
  }
  if (!state.hasMask && !state.hasIndexes) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected indexes operand or maskPattern for tscatter");
  }
  return success();
}

static ParseResult parseCommGroupOperands(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &groupOps) {
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
  return parser.parseRParen();
}

struct CommRecvClause {
  OpAsmParser::UnresolvedOperand ping;
  std::optional<OpAsmParser::UnresolvedOperand> pong;
  Type pingTy;
  Type pongTy;
};

static ParseResult parseCommCollectiveTypes(
    OpAsmParser &parser, SmallVectorImpl<Type> &fixedTypes,
    CommRecvClause &recvClause,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &groupOps,
    SmallVectorImpl<Type> &groupTypes) {
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
  if (recvClause.pong && (parser.parseComma() || parser.parseType(recvClause.pongTy)))
    return failure();
  for (size_t i = 0; i < groupOps.size(); ++i) {
    Type groupTy;
    if (parser.parseComma() || parser.parseType(groupTy))
      return failure();
    groupTypes.push_back(groupTy);
  }
  return parser.parseRParen();
}

static ParseResult parseRequiredCommAttrs(OpAsmParser &parser,
                                          OperationState &result,
                                          ArrayRef<StringRef> requiredAttrs) {
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
  return success();
}

static ParseResult resolveCommCollectiveOperands(
    OpAsmParser &parser, OperationState &result,
    ArrayRef<OpAsmParser::UnresolvedOperand> fixedOperands,
    SmallVectorImpl<Type> &fixedTypes, CommRecvClause &recvClause,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &groupOps,
    SmallVectorImpl<Type> &groupTypes) {
  for (auto [operand, type] : llvm::zip_equal(fixedOperands, fixedTypes)) {
    if (parser.resolveOperand(operand, type, result.operands))
      return failure();
  }
  if (parser.resolveOperand(recvClause.ping, recvClause.pingTy, result.operands))
    return failure();
  if (recvClause.pong &&
      parser.resolveOperand(*recvClause.pong, recvClause.pongTy, result.operands)) {
    return failure();
  }
  return parser.resolveOperands(groupOps, groupTypes, parser.getCurrentLocation(),
                                result.operands);
}

static void addCommCollectiveSegmentSizes(OpAsmParser &parser,
                                          OperationState &result,
                                          ArrayRef<int32_t> prefix,
                                          size_t groupCount) {
  SmallVec5<int32_t> segmentSizes(prefix.begin(), prefix.end());
  segmentSizes.push_back(static_cast<int32_t>(groupCount));
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(segmentSizes));
}

static ParseResult parsePartitionViewHeader(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand &source,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &offsets,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &sizes, Type &sourceTy) {
  if (parser.parseOperand(source) || parser.parseComma() ||
      parser.parseKeyword("offsets") || parser.parseEqual() ||
      parser.parseLSquare() || parser.parseOperandList(offsets) ||
      parser.parseRSquare() || parser.parseComma() ||
      parser.parseKeyword("sizes") || parser.parseEqual() ||
      parser.parseLSquare() || parser.parseOperandList(sizes) ||
      parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(sourceTy)) {
    return failure();
  }
  return success();
}

static void setPartitionViewOperandSegments(OperationState &result,
                                            size_t offsetsSize,
                                            size_t sizesSize) {
  auto &properties =
      result.getOrAddProperties<mlir::pto::PartitionViewOp::Properties>();
  llvm::copy(ArrayRef<int32_t>({1, static_cast<int32_t>(offsetsSize),
                                static_cast<int32_t>(sizesSize)}),
             properties.operandSegmentSizes.begin());
}

} // namespace

ParseResult mlir::pto::TGatherOp::parse(OpAsmParser &parser, OperationState &result) {
  TGatherParseState state;
  if (parseTGatherInsClause(parser, result, state) ||
      parseTGatherOutsClause(parser, state) ||
      parseOptionalTGatherMaskPattern(parser, result, state) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      validateTGatherOperands(parser, state) ||
      resolveTGatherOperands(parser, result, state)) {
    return failure();
  }
  addTGatherSegmentSizes(parser, result, state);
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
  TScatterParseState state;
  if (parseTScatterInsClause(parser, result, state) ||
      parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(state.dst) || parser.parseColonType(state.dstTy) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  if (result.attributes.get("maskPattern"))
    state.hasMask = true;
  if (validateTScatterOperands(parser, state) ||
      parser.resolveOperand(state.src, state.srcTy, result.operands) ||
      parser.resolveOperand(state.dst, state.dstTy, result.operands)) {
    return failure();
  }
  if (state.hasIndexes &&
      parser.resolveOperand(state.indexes, state.idxTy, result.operands)) {
    return failure();
  }
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
  if (parseCommGroupOperands(parser, groupOps) ||
      parseCommCollectiveTypes(parser, fixedTypes, recvClause, groupOps,
                               groupTypes) ||
      parseRequiredCommAttrs(parser, result, requiredAttrs) ||
      resolveCommCollectiveOperands(parser, result, fixedOperands, fixedTypes,
                                    recvClause, groupOps, groupTypes)) {
    return failure();
  }
  addCommCollectiveSegmentSizes(parser, result, operandSegmentsPrefix,
                                groupOps.size());
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
  SmallVec4<OpAsmParser::UnresolvedOperand> groupOps;
  SmallVec4<Type> groupTypes;
  if (parser.parseLParen() || parser.parseOperand(src) || parser.parseComma())
    return failure();
  if (failed(parseCommRecvClause(parser, recvClause)))
    return failure();

  SmallVec1<OpAsmParser::UnresolvedOperand> fixedOperands{src};
  SmallVec1<Type> fixedTypes(1);
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
  SmallVec4<OpAsmParser::UnresolvedOperand> groupOps;
  SmallVec4<Type> groupTypes;
  if (parser.parseLParen() || parser.parseOperand(dst) || parser.parseComma())
    return failure();
  if (failed(parseCommRecvClause(parser, recvClause)))
    return failure();

  SmallVec1<OpAsmParser::UnresolvedOperand> fixedOperands{dst};
  SmallVec1<Type> fixedTypes(1);
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
  SmallVec4<OpAsmParser::UnresolvedOperand> groupOps;
  SmallVec4<Type> groupTypes;
  if (parser.parseLParen() || parser.parseOperand(src) || parser.parseComma())
    return failure();
  if (failed(parseCommRecvClause(parser, recvClause)))
    return failure();

  SmallVec1<OpAsmParser::UnresolvedOperand> fixedOperands{src};
  SmallVec1<Type> fixedTypes(1);
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
  SmallVec4<OpAsmParser::UnresolvedOperand> groupOps;
  SmallVec4<Type> groupTypes;
  if (parser.parseLParen() || parser.parseOperand(dst) || parser.parseComma() ||
      parser.parseOperand(acc) || parser.parseComma())
    return failure();
  if (failed(parseCommRecvClause(parser, recvClause)))
    return failure();

  SmallVec2<OpAsmParser::UnresolvedOperand> fixedOperands{dst, acc};
  SmallVec2<Type> fixedTypes(kCommFixedOperandCount);
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
  SmallVec4<OpAsmParser::UnresolvedOperand> shapeOps;
  SmallVec4<OpAsmParser::UnresolvedOperand> strideOps;

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
      {1, static_cast<int32_t>(shapeOps.size()),
       static_cast<int32_t>(strideOps.size())});
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
  if (sizes.size() != static_cast<size_t>(sourceType.getRank()))
    return failure();

  SmallVec4<int64_t> shape;
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

static ParseResult parseOptionalArrowTypeAndResolveSource(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand &source, Type sourceTy, Type &resultTy,
    bool &hasExplicitResultTy) {
  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseType(resultTy))
      return failure();
    hasExplicitResultTy = true;
  }

  return parser.resolveOperand(source, sourceTy, result.operands);
}

static ParseResult resolveIndexOperandsToResult(
    OpAsmParser &parser, ArrayRef<OpAsmParser::UnresolvedOperand> operands,
    OperationState &result) {
  return parser.resolveOperands(operands, parser.getBuilder().getIndexType(),
                                result.operands);
}

ParseResult mlir::pto::PartitionViewOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  SmallVec4<OpAsmParser::UnresolvedOperand> offsets;
  SmallVec4<OpAsmParser::UnresolvedOperand> sizes;
  Type sourceTy;
  Type resultTy;
  bool hasExplicitResultTy = false;
  if (parsePartitionViewHeader(parser, result, source, offsets, sizes, sourceTy))
    return failure();
  if (parseOptionalArrowTypeAndResolveSource(parser, result, source, sourceTy,
                                             resultTy, hasExplicitResultTy))
    return failure();
  if (resolveIndexOperandsToResult(parser, offsets, result) ||
      resolveIndexOperandsToResult(parser, sizes, result))
    return failure();

  setPartitionViewOperandSegments(result, offsets.size(), sizes.size());
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

#include "PTOVerifyCore.cpp"
#include "PTOVerifyArithmeticA.cpp"
#include "PTOVerifyArithmeticB.cpp"
#include "PTOVerifyArithmeticC.cpp"
#include "PTOVerifyMisc.cpp"
#include "PTOParseAndSubview.cpp"
#include "PTOSubViewAndEffects.cpp"
#include "PTOEffectsExtra.cpp"

#include "PTO/IR/PTOInterfaces.cpp.inc"
#define GET_OP_CLASSES
#include "PTO/IR/PTOOps.cpp.inc"

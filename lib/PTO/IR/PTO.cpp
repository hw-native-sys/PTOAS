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
#include "llvm/Support/MathExtras.h"
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
static LogicalResult verifyVecTileStorage(Operation *op, Type ty,
                                          StringRef name);
static LogicalResult verifyNDStyleVecTile(Operation *op, Type ty,
                                          StringRef name,
                                          bool allowLowPrecision = false);
static LogicalResult verifyColReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy,
                                                   bool requireNonZeroSrc);
static LogicalResult verifyColArgReductionDstLayout(Operation *op, Type ty,
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
                                           Type dstTy,
                                           bool allowLowPrecision = false);
static LogicalResult verifyMatTileOperandsA2A3(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy,
                                               bool allowLowPrecision = false);
static LogicalResult verifyMatTileOperandsA5(Operation *op, Type lhsTy,
                                             Type rhsTy, Type dstTy,
                                             bool allowLowPrecision = false);
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

func::FuncOp mlir::pto::lookupPeerFuncAcrossContainer(Operation *op,
                                                      FlatSymbolRefAttr peerAttr) {
  if (!op || !peerAttr)
    return {};

  auto currentFunc = op->getParentOfType<func::FuncOp>();
  if (!currentFunc)
    return {};

  auto currentChildModule = currentFunc->getParentOfType<ModuleOp>();
  if (!currentChildModule)
    return {};

  StringRef target = peerAttr.getValue();
  for (func::FuncOp funcOp : currentChildModule.getOps<func::FuncOp>()) {
    if (funcOp.getSymName() == target)
      return funcOp;
  }
  if (auto localPeer = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupSymbolIn(currentChildModule, target))) {
    return localPeer;
  }

  Operation *maybeOuter = currentChildModule->getParentOp();
  auto outerModule = dyn_cast_or_null<ModuleOp>(maybeOuter);
  if (!outerModule)
    return {};

  SmallVector<func::FuncOp> fallbackMatches;
  outerModule.walk([&](func::FuncOp funcOp) {
    auto visibility = funcOp->getAttrOfType<StringAttr>("sym_visibility");
    if (visibility && visibility.getValue() == "private")
      return WalkResult::advance();

    StringRef symbolName = funcOp.getSymName();
    if (symbolName == target ||
        (funcOp->hasAttr(kPTODSLLogicalNameAttrName) &&
         getPTODSLLogicalNameOrSymbolName(funcOp) == target))
      fallbackMatches.push_back(funcOp);
    return WalkResult::advance();
  });

  if (fallbackMatches.size() == 1)
    return fallbackMatches.front();
  return {};
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
  }
  llvm_unreachable("unhandled pointer address space");
}

// Implementation is split into codecheck-sized fragments while retaining
// one translation unit and the original declaration order.
#include "Parts/PTOOpsPart01.cpp"
#include "Parts/PTOOpsPart02.cpp"
#include "Parts/PTOOpsPart03.cpp"
#include "Parts/PTOOpsPart04.cpp"
#include "Parts/PTOOpsPart05.cpp"
#include "Parts/PTOOpsPart06.cpp"
#include "Parts/PTOOpsPart07.cpp"
#include "Parts/PTOOpsPart08.cpp"
#include "Parts/PTOOpsPart09.cpp"
#include "Parts/PTOOpsPart10.cpp"

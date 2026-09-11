// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This implementation fragment is included by PTO.cpp and intentionally is
// not listed as a separate CMake translation unit.


// Forward declarations for custom shape/type printers used by tensor_view and
// partition_tensor_view.
namespace mlir {
namespace pto {
static LogicalResult parseShapeAndElem(AsmParser &parser,
                                       SmallVectorImpl<int64_t> &shape,
                                       Type &elementType,
                                       bool allowDynamic = true);
static LogicalResult parseViewShapeElemAndLayout(
    AsmParser &parser, SmallVectorImpl<int64_t> &shape, Type &elementType,
    Attribute &layout, bool allowDynamic = true);
static void printShapeAndElem(AsmPrinter &printer,
                              ArrayRef<int64_t> shape,
                              Type elementType);
static void printViewShapeElemAndLayout(AsmPrinter &printer,
                                        ArrayRef<int64_t> shape,
                                        Type elementType, Attribute layout);
} // namespace pto
} // namespace mlir

// =============================================================================
// TileBufType 的自定义 Shape 解析与打印函数
// =============================================================================

// 解析逻辑：解析形如 "32x32" 的维度列表
[[maybe_unused]] static ParseResult parseShape(AsmParser &parser, SmallVectorImpl<int64_t> &shape) {
  // parseDimensionList 会解析 "dim x dim x ...", 遇到无法解析为维度的字符停止
  // 参数 allowDynamic=true (允许 ?), withTrailingX=false (不吞掉末尾的 x)
  if (parser.parseDimensionList(shape, /*allowDynamic=*/true, /*withTrailingX=*/false)) {
    return failure();
  }
  return success();
}

// 打印逻辑：打印形如 "32x32" 的维度列表
[[maybe_unused]] static void printShape(AsmPrinter &printer, ArrayRef<int64_t> shape) {
  for (auto it = shape.begin(); it != shape.end(); ++it) {
    if (it != shape.begin()) {
      printer << "x"; // 维度间的分隔符
    }
    if (*it == ShapedType::kDynamic) {
      printer << "?";
    } else {
      printer << *it;
}
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
static LogicalResult verifyNoUnpublishedFixpipeFrontendAttrs(Operation *op);
static ParseResult parseSyncEventOpCommon(OpAsmParser &parser,
                                          OperationState &result,
                                          StringAttr pipeAttrName,
                                          StringAttr eventIdAttrName);
static void printSyncEventOpCommon(OpAsmPrinter &p, Operation *op,
                                   PipeAttr pipeAttr, IntegerAttr eventAttr,
                                   Value eventDyn, StringRef pipeAttrName,
                                   StringRef eventIdAttrName);
static bool isTileLikeType(Type ty);
static SmallVector<int64_t, mlir::pto::kValue4> getShapeVec(Type ty);
static SmallVector<int64_t, mlir::pto::kValue4> getValidShapeVec(Type ty);
static SmallVector<int64_t, 4> getValidShapeVec(Value value);
static bool isKnownZeroOrUnitExtent(int64_t value);
static bool isByteIntegerType(Type ty);
static LogicalResult verifyTileBufCommon(Operation *op, Type ty, StringRef name,
                                         bool allowLowPrecision = false);
static LogicalResult verifyTmpCapacityAtLeast(Operation *op, Type tmpTy,
                                              uint64_t requiredBytes,
                                              StringRef tmpName = "tmp");

namespace {
struct PTOInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
};
} // namespace
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
  if (failed(parser.parseLess())) {
    return failure();
  }

  if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/true))) {
    return failure();
  }

  if (failed(parser.parseType(elementType))) {
    return failure();
  }

  if (failed(parser.parseGreater())) {
    return failure();
  }

  return success();
}

static int64_t getPTOTypeRank(Type type) {
  // 1. 处理标准的 MLIR 类型 (Tensor, Vector)
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    if (shapedTy.hasRank()) {
      return shapedTy.getRank();
    }
    return -1; // Unranked type
  }

  // 2. 处理 PTO 自定义类型
  if (auto tvTy = dyn_cast<pto::TensorViewType>(type)) {
    return tvTy.getRank();
  }

  if (auto tileTy = dyn_cast<pto::TileType>(type)) {
    return tileTy.getRank();
  }

  if (auto tileViewTy = dyn_cast<pto::PartitionTensorViewType>(type)) {
    return tileViewTy.getRank();
  }

  if (auto tileBufTy = dyn_cast<pto::TileBufType>(type)) {
    return tileBufTy.getRank();
  }

  // 3. 不支持的类型
  return -1;
}

func::FuncOp mlir::pto::lookupPeerFuncAcrossContainer(Operation *op,
                                                      FlatSymbolRefAttr peerAttr) {
  if (!op || !peerAttr) {
    return {};
  }

  auto currentFunc = op->getParentOfType<func::FuncOp>();
  if (!currentFunc) {
    return {};
  }

  auto currentChildModule = currentFunc->getParentOfType<ModuleOp>();
  if (!currentChildModule) {
    return {};
  }

  StringRef target = peerAttr.getValue();
  for (func::FuncOp funcOp : currentChildModule.getOps<func::FuncOp>()) {
    if (funcOp.getSymName() == target) {
      return funcOp;
    }
  }
  if (auto localPeer = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupSymbolIn(currentChildModule, target))) {
    return localPeer;
  }

  Operation *maybeOuter = currentChildModule->getParentOp();
  auto outerModule = dyn_cast_or_null<ModuleOp>(maybeOuter);
  if (!outerModule) {
    return {};
  }

  SmallVector<func::FuncOp> fallbackMatches;
  outerModule.walk([&](func::FuncOp funcOp) {
    auto visibility = funcOp->getAttrOfType<StringAttr>("sym_visibility");
    if (visibility && visibility.getValue() == "private") {
      return WalkResult::advance();
    }

    StringRef symbolName = funcOp.getSymName();
    if (symbolName == target ||
        (funcOp->hasAttr(kPTODSLLogicalNameAttrName) &&
         getPTODSLLogicalNameOrSymbolName(funcOp) == target)) {
      fallbackMatches.push_back(funcOp);
    }
    return WalkResult::advance();
  });

  if (fallbackMatches.size() == 1) {
    return fallbackMatches.front();
  }
  return {};
}

static bool isA5DeviceSpec(StringRef spec) {
  return spec.starts_with("Ascend950") || spec.starts_with("Ascend910_95");
}

static bool isA5ModuleTarget(ModuleOp module) {
  if (!module) {
    return false;
  }
  if (auto arch = module->getAttrOfType<StringAttr>(kPTOTargetArchAttrName)) {
    if (arch.getValue().equals_insensitive("a5")) {
      return true;
    }
  }
  if (auto spec = module->getAttrOfType<StringAttr>("pto.device-spec")) {
    return isA5DeviceSpec(spec.getValue());
  }
  return false;
}

PTOArch mlir::pto::getTargetArch(ModuleOp module) {
  if (isA5ModuleTarget(module)) {
    return PTOArch::A5;
  }

  switch (getPTOParserTargetArch(module ? module.getContext() : nullptr)) {
  case PTOParserTargetArch::A5:
    return PTOArch::A5;
  case PTOParserTargetArch::A3:
  case PTOParserTargetArch::Unspecified:
    break;
  }
  return PTOArch::A3;
}

PTOArch mlir::pto::getTargetArch(Operation *op) {
  if (!op) {
    return PTOArch::A3;
  }
  if (auto module = op->getParentOfType<ModuleOp>()) {
    return getTargetArch(module);
  }
  switch (getPTOParserTargetArch(op->getContext())) {
  case PTOParserTargetArch::A5:
    return PTOArch::A5;
  case PTOParserTargetArch::A3:
  case PTOParserTargetArch::Unspecified:
    break;
  }
  return PTOArch::A3;
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

constexpr int64_t kA5VectorLengthBytes = 256;

enum class PredicateLoadDist {
  Norm,
  Us,
  Ds,
};

enum class PredicateStoreDist {
  Norm,
  Pk,
};

struct PredicateLoadAlignmentRule {
  PredicateLoadDist dist;
  int64_t alignmentBytes;
};

struct PredicateStoreAlignmentRule {
  PredicateStoreDist dist;
  int64_t alignmentBytes;
};

constexpr PredicateLoadAlignmentRule kA5PredicateLoadAlignmentRules[] = {
    {PredicateLoadDist::Norm, kA5VectorLengthBytes / 8},
    {PredicateLoadDist::Us, kA5VectorLengthBytes / 16},
    {PredicateLoadDist::Ds, std::min<int64_t>(32, kA5VectorLengthBytes / 4)},
};

constexpr PredicateStoreAlignmentRule kA5PredicateStoreAlignmentRules[] = {
    {PredicateStoreDist::Norm, kA5VectorLengthBytes / 8},
    {PredicateStoreDist::Pk, kA5VectorLengthBytes / 16},
};

static std::optional<PredicateLoadDist>
parsePredicateLoadDist(StringRef dist) {
  if (dist == "NORM") {
    return PredicateLoadDist::Norm;
  }
  if (dist == "US") {
    return PredicateLoadDist::Us;
  }
  if (dist == "DS") {
    return PredicateLoadDist::Ds;
  }
  return std::nullopt;
}

static std::optional<PredicateStoreDist>
parsePredicateStoreDist(StringRef dist) {
  if (dist == "NORM") {
    return PredicateStoreDist::Norm;
  }
  if (dist == "PK") {
    return PredicateStoreDist::Pk;
  }
  return std::nullopt;
}

template <typename Rule, typename Dist, size_t N>
static std::optional<int64_t> findAlignmentSize(const Rule (&rules)[N],
                                                Dist dist) {
  auto rule = llvm::find_if(
      rules, [&](const Rule &entry) { return entry.dist == dist; });
  if (rule == std::end(rules)) {
    return std::nullopt;
  }
  return rule->alignmentBytes;
}

std::optional<int64_t>
mlir::pto::getLoadStoreVecAlignmentSize(Operation *op) {
  if (!op || getTargetArch(op) != PTOArch::A5) {
    return std::nullopt;
  }

  if (auto pldi = dyn_cast<PldiOp>(op)) {
    auto dist = parsePredicateLoadDist(pldi.getDist());
    return dist ? findAlignmentSize(kA5PredicateLoadAlignmentRules, *dist)
                : std::nullopt;
  }
  if (auto psti = dyn_cast<PstiOp>(op)) {
    auto dist = parsePredicateStoreDist(psti.getDist());
    return dist ? findAlignmentSize(kA5PredicateStoreAlignmentRules, *dist)
                : std::nullopt;
  }
  if (auto sprsti = dyn_cast<SprstiOp>(op)) {
    if (sprsti.getSpr() == "AR") {
        return mlir::pto::kValue4;
    }
  }
  return std::nullopt;
}

static llvm::TypeSize getOneByteTypeSize() { return llvm::TypeSize::getFixed(mlir::pto::kValue8); }

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

static Type parseShapedPTOType(OpAsmParser& parser, StringRef head)
{
    SmallVector<int64_t, mlir::pto::kValue4> shape;
    Type elementType;
    if (failed(parsePTOShapeAndElement(parser, shape, elementType))) {
        return {};
    }
    MLIRContext* context = parser.getContext();
    if (head == "pto.tile_view") {
        return PartitionTensorViewType::get(context, shape, elementType);
    }
    if (head == "pto.tile") {
        return TileType::get(context, shape, elementType);
    }
    return TensorViewType::get(context, shape, elementType);
}

static Type parsePtrPTOType(OpAsmParser& parser)
{
    Type elementType;
    if (parser.parseLess() || parser.parseType(elementType)) {
        return {};
    }
    MLIRContext* context = parser.getContext();
    auto memorySpace = AddressSpaceAttr::get(context, AddressSpace::GM);
    if (succeeded(parser.parseOptionalComma())) {
        StringRef keyword;
        if (parser.parseKeyword(&keyword)) {
            return {};
        }
        auto parsed = parsePtrAddressSpaceKeyword(keyword);
        if (!parsed) {
            parser.emitError(
                parser.getCurrentLocation(), "!pto.ptr address space must be one of "
                                             "`gm|ub|mat|l1|left|l0a|right|l0b|acc|l0c|vec|bias|bt|scaling|fb`");
            return {};
        }
        memorySpace = AddressSpaceAttr::get(context, *parsed);
    }
    if (parser.parseGreater()) {
        return {};
    }
    return PtrType::get(context, elementType, memorySpace);
}

static Type parseKnownPTOType(OpAsmParser& parser, StringRef head)
{
    if (head == "pto.ptr") {
        return parsePtrPTOType(parser);
    }
    if (head == "pto.tile_view" || head == "pto.tile" || head == "pto.tensor_view") {
        return parseShapedPTOType(parser, head);
    }
    return {};
}

[[maybe_unused]] static Type parsePTOTypeAllowNoBang(OpAsmParser& parser)
{
    Type type;
    OptionalParseResult optionalType = parser.parseOptionalType(type);
    if (optionalType.has_value()) {
        return failed(*optionalType) ? Type() : type;
    }
    StringRef head;
    if (parser.parseKeyword(&head)) {
        return {};
    }
    return parseKnownPTOType(parser, head);
}

mlir::Type TensorViewType::parse(::mlir::AsmParser& parser)
{
    SmallVector<int64_t, mlir::pto::kValue4> shape;
    Type elementType;
    Attribute layout;
    if (failed(parseViewShapeElemAndLayout(
            parser, shape, elementType, layout, /*allowDynamic=*/true))) {
        return Type();
    }
    return TensorViewType::get(parser.getContext(), shape, elementType, layout);
}

void TensorViewType::print(::mlir::AsmPrinter& printer) const
{
    printViewShapeElemAndLayout(printer, getShape(), getElementType(),
                                getLayout());
}

mlir::Type PtrType::parse(::mlir::AsmParser& parser)
{
    Type elementType;
    if (failed(parser.parseLess()) || failed(parser.parseType(elementType))) {
        return {};
    }

    auto memorySpace = pto::AddressSpaceAttr::get(parser.getContext(), pto::AddressSpace::GM);
    if (succeeded(parser.parseOptionalComma())) {
        StringRef memorySpaceKeyword;
        if (failed(parser.parseKeyword(&memorySpaceKeyword))) {
            return {};
        }
        auto parsed = parsePtrAddressSpaceKeyword(memorySpaceKeyword);
        if (!parsed) {
            parser.emitError(
                parser.getCurrentLocation(), "!pto.ptr address space must be one of "
                                             "`gm|ub|mat|l1|left|l0a|right|l0b|acc|l0c|vec|bias|bt|scaling|fb`");
            return {};
        }
        memorySpace = pto::AddressSpaceAttr::get(parser.getContext(), *parsed);
    }

    if (failed(parser.parseGreater())) {
        return {};
    }
    return PtrType::get(parser.getContext(), elementType, memorySpace);
}

void PtrType::print(::mlir::AsmPrinter& printer) const
{
    printer << "<" << getElementType();
    StringRef memorySpaceKeyword = printPtrAddressSpaceKeyword(getMemorySpace().getAddressSpace());
    if (!memorySpaceKeyword.empty()) {
        printer << ", " << memorySpaceKeyword;
    }
    printer << ">";
}

//===----------------------------------------------------------------------===//
// pto.tdivs custom asm to support both:
//   pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
//   pto.tdivs ins(%scalar, %src : f32, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
// The operand order in the op follows textual input order.
//===----------------------------------------------------------------------===//

static ParseResult parseTDivSOperands(
    OpAsmParser& parser, OpAsmParser::UnresolvedOperand& op0, OpAsmParser::UnresolvedOperand& op1, Type& ty0, Type& ty1)
{
    if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(op0) || parser.parseComma() ||
        parser.parseOperand(op1) || parser.parseColonType(ty0) || parser.parseComma() || parser.parseType(ty1) ||
        parser.parseRParen()) {
        return failure();
    }
    return success();
}

static ParseResult parseTDivSResult(OpAsmParser& parser, OpAsmParser::UnresolvedOperand& dst, Type& dstType)
{
    if (parser.parseKeyword("outs") || parser.parseLParen() || parser.parseOperand(dst) ||
        parser.parseColonType(dstType) || parser.parseRParen()) {
        return failure();
    }
    return success();
}

static ParseResult validateTDivSTypes(OpAsmParser& parser, Type ty0, Type ty1, Type dstType)
{
    auto tile0 = dyn_cast<mlir::pto::TileBufType>(ty0);
    auto tile1 = dyn_cast<mlir::pto::TileBufType>(ty1);
    if ((tile0 && tile1) || (!tile0 && !tile1)) {
        return parser.emitError(
            parser.getCurrentLocation(), "expected exactly one tile_buf operand and one scalar operand");
    }

    if (!dyn_cast<mlir::pto::TileBufType>(dstType)) {
        return parser.emitError(parser.getCurrentLocation(), "expected outs type to be !pto.tile_buf<...>");
    }
    return success();
}

static ParseResult resolveTDivSOperands(
    OpAsmParser& parser, OperationState& result, OpAsmParser::UnresolvedOperand op0, OpAsmParser::UnresolvedOperand op1,
    OpAsmParser::UnresolvedOperand dst, Type ty0, Type ty1, Type dstType)
{
    if (parser.resolveOperand(op0, ty0, result.operands) || parser.resolveOperand(op1, ty1, result.operands) ||
        parser.resolveOperand(dst, dstType, result.operands)) {
        return failure();
    }
    return success();
}

ParseResult mlir::pto::TDivSOp::parse(OpAsmParser& parser, OperationState& result)
{
    OpAsmParser::UnresolvedOperand op0, op1, dst;
    Type ty0, ty1, dstType;
    NamedAttrList attrs;
    if (parseTDivSOperands(parser, op0, op1, ty0, ty1) || parseTDivSResult(parser, dst, dstType) ||
        parser.parseOptionalAttrDict(attrs) || validateTDivSTypes(parser, ty0, ty1, dstType) ||
        resolveTDivSOperands(parser, result, op0, op1, dst, ty0, ty1, dstType)) {
        return failure();
    }

    result.addAttributes(attrs);
    return success();
}

void mlir::pto::TDivSOp::print(OpAsmPrinter& p)
{
    p << " ins(";
    p << getSrc() << ", " << getScalar() << " : " << getSrc().getType() << ", " << getScalar().getType();
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

#include "PTOGatherScatterParser.cpp"

void mlir::pto::TGatherOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", ";
  if (auto mp = getMaskPatternAttr()) {
    p << "{maskPattern = " << mp << "} : " << getSrc().getType();
    if (auto axisAttr = getAxisAttr()) {
      p << ", " << axisAttr;
    }
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
  if (getCdst()) {
    p << ", " << getCdst();
  }
  p << " : " << getDst().getType();
  if (getCdst()) {
    p << ", " << getCdst().getType();
  }
  p << ")";

  if (getMaskPatternAttr()) {
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/{"maskPattern", "axis", "operandSegmentSizes"});
  } else {
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/{"axis", "operandSegmentSizes"});
  }
}


void mlir::pto::TScatterOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", ";
  if (getMaskPatternAttr()) {
    p << "{maskPattern = " << getMaskPatternAttr() << "} : " << getSrc().getType();
    if (auto axisAttr = getAxisAttr()) {
      p << ", " << axisAttr;
    }
  } else {
    p << getIndexes() << " : " << getSrc().getType() << ", "
      << getIndexes().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"maskPattern", "axis"});
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
      parser.parseOperand(recvClause.ping)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand pong;
    if (parser.parseOperand(pong)) {
      return failure();
    }
    recvClause.pong = pong;
  }
  return parser.parseRParen();
}

static ParseResult parseCommGroupOperands(
    OpAsmParser& parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand>& groupOps)
{
    if (parser.parseComma() || parser.parseKeyword("group") || parser.parseLParen()) {
        return failure();
    }

    OpAsmParser::UnresolvedOperand group;
    if (parser.parseOperand(group)) {
        return failure();
    }
    groupOps.push_back(group);
    while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseOperand(group)) {
            return failure();
        }
        groupOps.push_back(group);
    }

    if (parser.parseRParen()) {
        return failure();
    }

    return success();
}

static ParseResult parseCommOperandTypes(
    OpAsmParser& parser, SmallVectorImpl<Type>& fixedTypes, CommRecvClause& recvClause, size_t groupCount,
    SmallVectorImpl<Type>& groupTypes)
{
    if (parser.parseColon()) {
        return failure();
    }

    for (size_t i = 0; i < fixedTypes.size(); ++i) {
        if (i != 0 && parser.parseComma()) {
            return failure();
        }
        if (parser.parseType(fixedTypes[i])) {
            return failure();
        }
    }
    if (parser.parseComma() || parser.parseType(recvClause.pingTy)) {
        return failure();
    }
    if (recvClause.pong) {
        if (parser.parseComma() || parser.parseType(recvClause.pongTy)) {
            return failure();
        }
    }
    for (size_t i = 0; i < groupCount; ++i) {
        Type groupTy;
        if (parser.parseComma() || parser.parseType(groupTy)) {
            return failure();
        }
        groupTypes.push_back(groupTy);
    }
    if (parser.parseRParen()) {
        return failure();
    }

    return success();
}

static ParseResult parseCommRequiredAttributes(
    OpAsmParser& parser, OperationState& result, ArrayRef<StringRef> requiredAttrs)
{
    NamedAttrList attrs;
    if (parser.parseOptionalAttrDict(attrs)) {
        return failure();
    }
    for (StringRef attrName : requiredAttrs) {
        if (!attrs.get(attrName)) {
            return parser.emitError(parser.getCurrentLocation()) << "expected '" << attrName << "' attribute";
        }
    }
    result.addAttributes(attrs);

    return success();
}

static ParseResult resolveCommOperands(
    OpAsmParser& parser, OperationState& result, ArrayRef<OpAsmParser::UnresolvedOperand> fixedOperands,
    ArrayRef<Type> fixedTypes, CommRecvClause& recvClause, ArrayRef<OpAsmParser::UnresolvedOperand> groupOps,
    ArrayRef<Type> groupTypes)
{
    for (auto [operand, type] : llvm::zip_equal(fixedOperands, fixedTypes)) {
        if (parser.resolveOperand(operand, type, result.operands)) {
            return failure();
        }
    }
    if (parser.resolveOperand(recvClause.ping, recvClause.pingTy, result.operands)) {
        return failure();
    }
    if (recvClause.pong && parser.resolveOperand(*recvClause.pong, recvClause.pongTy, result.operands)) {
        return failure();
    }
    if (parser.resolveOperands(groupOps, groupTypes, parser.getCurrentLocation(), result.operands)) {
        return failure();
    }

    return success();
}

static ParseResult parseCommCollectiveTail(
    OpAsmParser& parser, OperationState& result, ArrayRef<OpAsmParser::UnresolvedOperand> fixedOperands,
    SmallVectorImpl<Type>& fixedTypes, CommRecvClause& recvClause,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& groupOps, SmallVectorImpl<Type>& groupTypes,
    ArrayRef<int32_t> operandSegmentsPrefix, ArrayRef<StringRef> requiredAttrs)
{
    if (parseCommGroupOperands(parser, groupOps) ||
        parseCommOperandTypes(parser, fixedTypes, recvClause, groupOps.size(), groupTypes) ||
        parseCommRequiredAttributes(parser, result, requiredAttrs) ||
        resolveCommOperands(parser, result, fixedOperands, fixedTypes, recvClause, groupOps, groupTypes)) {
        return failure();
    }
    SmallVector<int32_t, mlir::pto::kValue5> segmentSizes(operandSegmentsPrefix.begin(), operandSegmentsPrefix.end());
    segmentSizes.push_back(static_cast<int32_t>(groupOps.size()));
    result.addAttribute("operandSegmentSizes", parser.getBuilder().getDenseI32ArrayAttr(segmentSizes));
    return success();
}

static void printCommRecvClause(OpAsmPrinter& p, Value ping, Value pong)
{
    p << "recv(" << ping;
    if (pong) {
        p << ", " << pong;
    }
    p << ")";
}

static void printCommGroupTypes(OpAsmPrinter& p, ValueRange group)
{
    for (Value groupValue : group) {
        p << ", " << groupValue.getType();
    }
}

static void printCommGroupClause(OpAsmPrinter& p, ValueRange group)
{
    p << "group(";
    p.printOperands(group);
    p << ")";
}

} // namespace

static ParseResult parseCommSingleFixedOperand(OpAsmParser& parser, OperationState& result)
{
    OpAsmParser::UnresolvedOperand src;
    CommRecvClause recvClause;
    SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue4> groupOps;
    SmallVector<Type, mlir::pto::kValue4> groupTypes;

    if (parser.parseLParen() || parser.parseOperand(src) || parser.parseComma()) {
        return failure();
    }
    if (failed(parseCommRecvClause(parser, recvClause))) {
        return failure();
    }

    SmallVector<OpAsmParser::UnresolvedOperand, 1> fixedOperands{src};
    SmallVector<Type, 1> fixedTypes(1);
    if (failed(parseCommCollectiveTail(
            parser, result, fixedOperands, fixedTypes, recvClause, groupOps, groupTypes,
            {1, 1, recvClause.pong ? 1 : 0}, {"root"}))) {
        return failure();
    }
    return success();
}

static void printCommSingleFixedOperand(
    OpAsmPrinter& p, Operation* op, Value fixed, Value ping, Value pong, ValueRange group)
{
    p << "(" << fixed << ", ";
    printCommRecvClause(p, ping, pong);
    p << ", ";
    printCommGroupClause(p, group);
    p << " : " << fixed.getType() << ", " << ping.getType();
    if (pong) {
        p << ", " << pong.getType();
    }
    printCommGroupTypes(p, group);
    p << ")";
    p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TBroadcastOp::parse(OpAsmParser& parser, OperationState& result)
{
    return parseCommSingleFixedOperand(parser, result);
}

void mlir::pto::TBroadcastOp::print(OpAsmPrinter& p)
{
    printCommSingleFixedOperand(p, getOperation(), getSrc(), getPing(), getPong(), getGroup());
}

ParseResult mlir::pto::CommTGatherOp::parse(OpAsmParser& parser, OperationState& result)
{
    return parseCommSingleFixedOperand(parser, result);
}

void mlir::pto::CommTGatherOp::print(OpAsmPrinter& p)
{
    printCommSingleFixedOperand(p, getOperation(), getDst(), getPing(), getPong(), getGroup());
}

ParseResult mlir::pto::CommTScatterOp::parse(OpAsmParser& parser, OperationState& result)
{
    return parseCommSingleFixedOperand(parser, result);
}

void mlir::pto::CommTScatterOp::print(OpAsmPrinter &p) {
  printCommSingleFixedOperand(p, getOperation(), getSrc(),
                             getPing(), getPong(), getGroup());
}

ParseResult mlir::pto::TReduceOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::UnresolvedOperand dst, acc;
  CommRecvClause recvClause;
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue4> groupOps;
  SmallVector<Type, mlir::pto::kValue4> groupTypes;

  if (parser.parseLParen() || parser.parseOperand(dst) || parser.parseComma() ||
      parser.parseOperand(acc) || parser.parseComma()) {
    return failure();
  }
  if (failed(parseCommRecvClause(parser, recvClause))) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue2> fixedOperands{dst, acc};
  SmallVector<Type, mlir::pto::kValue2> fixedTypes(mlir::pto::kValue2);
  if (failed(parseCommCollectiveTail(
          parser, result, fixedOperands, fixedTypes, recvClause, groupOps,
          groupTypes, {1, 1, 1, recvClause.pong ? 1 : 0},
          {"reduceOp", "root"}))) {
    return failure();
  }
  return success();
}

void mlir::pto::TReduceOp::print(OpAsmPrinter &p) {
  p << "(" << getDst() << ", " << getAcc() << ", ";
  printCommRecvClause(p, getRecvPing(), getRecvPong());
  p << ", ";
  printCommGroupClause(p, getGroup());
  p << " : " << getDst().getType() << ", " << getAcc().getType() << ", "
    << getRecvPing().getType();
  if (getRecvPong()) {
    p << ", " << getRecvPong().getType();
  }
  printCommGroupTypes(p, getGroup());
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::MakeTensorViewOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  OpAsmParser::UnresolvedOperand ptr;
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue4> shapeOps;
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue4> strideOps;

  Type resultTy;

  // %ptr
  if (parser.parseOperand(ptr)) {
    return failure();
  }

  // , shape = [ ... ]
  if (parser.parseComma() || parser.parseKeyword("shape") || parser.parseEqual() ||
      parser.parseLSquare() ||
      parser.parseOperandList(shapeOps) ||
      parser.parseRSquare()) {
    return failure();
  }

  // strides = [ ... ]
  if (parser.parseComma() || parser.parseKeyword("strides") || parser.parseEqual() ||
      parser.parseLSquare() ||
      parser.parseOperandList(strideOps) ||
      parser.parseRSquare()) {
    return failure();
  }

  // attr-dict
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // : result-type
  if (parser.parseColonType(resultTy)) {
    return failure();
  }
  result.addTypes(resultTy);

  auto tvTy = llvm::dyn_cast<mlir::pto::TensorViewType>(resultTy);
  if (!tvTy) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected result type pto.tensor_view<...>");
  }

  Type elemTy = tvTy.getElementType();

  Type ptrTy = mlir::pto::PtrType::get(parser.getContext(), elemTy);
  // resolve %ptr
  if (parser.resolveOperand(ptr, ptrTy, result.operands)) {
    return failure();
  }

  // resolve shape/strides 为 index
  Type indexTy = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(shapeOps, indexTy, result.operands)) {
    return failure();
  }
  if (parser.resolveOperands(strideOps, indexTy, result.operands)) {
    return failure();
  }

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
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
    return c.value();
  }
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue())) {
      return ia.getInt();
    }
  }
  return std::nullopt;
}

static FailureOr<mlir::pto::PartitionTensorViewType>
inferPartitionViewResultTypeFromSizes(Type sourceType, ValueRange sizes) {
  int64_t sourceRank = 0;
  Type elementType;
  Attribute layout;
  if (auto tensorView = dyn_cast<mlir::pto::TensorViewType>(sourceType)) {
    sourceRank = tensorView.getRank();
    elementType = tensorView.getElementType();
    layout = tensorView.getLayout();
  } else if (auto partitionView =
                 dyn_cast<mlir::pto::PartitionTensorViewType>(sourceType)) {
    sourceRank = partitionView.getRank();
    elementType = partitionView.getElementType();
    layout = partitionView.getLayout();
  } else {
    return failure();
  }

  if ((int64_t)sizes.size() != sourceRank) {
    return failure();
  }

  SmallVector<int64_t, mlir::pto::kValue4> shape;
  shape.reserve(sizes.size());
  for (Value size : sizes) {
    auto constSize = getConstIndexValue(size);
    if (constSize && *constSize >= 0) {
      shape.push_back(*constSize);
    } else {
      shape.push_back(ShapedType::kDynamic);
}
  }

  return mlir::pto::PartitionTensorViewType::get(sourceType.getContext(), shape,
                                                 elementType, layout);
}

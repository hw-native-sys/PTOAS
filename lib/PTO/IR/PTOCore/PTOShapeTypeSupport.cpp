// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

using pto::TMrgSortOp;
using pto::TMulOp;
using pto::TMulSOp;
using pto::TNegOp;
using pto::TNotifyOp;
using pto::TNotOp;
using pto::TOrOp;
using pto::TOrSOp;
using pto::TPartAddOp;
using pto::TPartArgMaxOp;
using pto::TPartArgMinOp;
using pto::TPartMaxOp;
using pto::TPartMinOp;
using pto::TPartMulOp;
using pto::TPopFromAicOp;
using pto::TPopFromAivOp;
using pto::TPopOp;
using pto::TPowOp;
using pto::TPowSOp;
using pto::TPrefetchAsyncOp;
using pto::TPrefetchOp;
using pto::TPReluOp;
using pto::TPrintOp;
using pto::TPushOp;
using pto::TPushToAicOp;
using pto::TPushToAivOp;
using pto::TPutAsyncOp;
using pto::TPutOp;
using pto::TQuantMxOp;
using pto::TQuantOp;
using pto::TRandomOp;
using pto::TRecipOp;
using pto::TReduceOp;
using pto::TReluOp;
using pto::TRemOp;
using pto::TRemSOp;
using pto::TReshapeOp;
using pto::TRowArgMaxOp;
using pto::TRowArgMinOp;
using pto::TRowExpandAddOp;
using pto::TRowExpandDivOp;
using pto::TRowExpandExpdifOp;
using pto::TRowExpandMaxOp;
using pto::TRowExpandMinOp;
using pto::TRowExpandMulOp;
using pto::TRowExpandOp;
using pto::TRowExpandSubOp;
using pto::TRowMaxOp;
using pto::TRowMinOp;
using pto::TRowProdOp;
using pto::TRowSumOp;
using pto::TRsqrtOp;
using pto::TScatterOp;
using pto::TSelOp;
using pto::TSelSOp;
using pto::TSetValOp;
using pto::TShlOp;
using pto::TShlSOp;
using pto::TShrOp;
using pto::TShrSOp;
using pto::TSort32Op;
using pto::TSqrtOp;
using pto::TStoreOp;
using pto::TSubCOp;
using pto::TSubOp;
using pto::TSubSCOp;
using pto::TSubSOp;
using pto::TTestOp;
using pto::TTransOp;
using pto::TTriOp;
using pto::TWaitOp;
using pto::TXorOp;
using pto::TXorSOp;
using pto::WaitAsyncEventOp;
using pto::WaitCrossBlockOp;
using pto::WaitFlagOp;
using pto::WaitIntraBlockOp;
using pto::YieldOp;

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

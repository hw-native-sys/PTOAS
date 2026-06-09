// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOToEmitC.cpp - PTO to EmitC conversion pass ----------------------===//
//===----------------------------------------------------------------------===//

#pragma GCC diagnostic ignored "-Woverloaded-virtual"
// https://discourse.llvm.org/t/matchandrewrite-hiding-virtual-functions/84933/8

#include <cassert>
#include <climits>

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTOToEmitCInternal.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"                   
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
#define GEN_PASS_DEF_EMITPTOMANUAL
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {
constexpr unsigned kInlineCapacity2 = 2;
constexpr unsigned kInlineCapacity3 = 3;
constexpr unsigned kInlineCapacity4 = 4;
constexpr unsigned kInlineCapacity5 = 5;
constexpr size_t kTileRank2D = 2;
constexpr unsigned kNumber1 = 1;
constexpr unsigned kNumber2 = 2;
constexpr unsigned kNumber4 = 4;

constexpr size_t kPaddedShapeInnerRowDim2 = 2;
constexpr size_t kPaddedShapeInnerColDim3 = 3;
constexpr size_t kPaddedShapeInnermostDim4 = 4;
constexpr int64_t kPaddedShapeUnitStride1 = 1;

template <typename T>
using SmallVec2 = SmallVector<T, kInlineCapacity2>;
template <typename T>
using SmallVec3 = SmallVector<T, kInlineCapacity3>;
template <typename T>
using SmallVec4 = SmallVector<T, kInlineCapacity4>;
template <typename T>
using SmallVec5 = SmallVector<T, kInlineCapacity5>;
} // namespace

static bool getStaticMemrefLayout(MemRefType mrTy,
                                  SmallVectorImpl<int64_t> &strides,
                                  int64_t &offset);
static std::string getGlobalTensorTypeStringFromShape(Type elemTy,
                                                      ArrayRef<int64_t> shape,
                                                      StringRef layoutEnum =
                                                          "pto::Layout::ND");
static emitc::OpaqueType getGlobalTensorOpaqueTypeFromShape(
    MLIRContext *ctx, Type elemTy, ArrayRef<int64_t> shape,
    StringRef layoutEnum = "pto::Layout::ND");

llvm::StringRef mlir::pto::addrSpaceQualifier(pto::AddressSpace as) {
  switch (as) {
  case pto::AddressSpace::Zero:
    return "__gm__";
  case pto::AddressSpace::VEC:
    return "__ubuf__";
  case pto::AddressSpace::GM:
    return "__gm__";
  case pto::AddressSpace::MAT:
    return "__cbuf__";
  case pto::AddressSpace::LEFT:
    return "__ca__";
  case pto::AddressSpace::RIGHT:
    return "__cb__";
  case pto::AddressSpace::ACC:
    return "__cc__";
  case pto::AddressSpace::BIAS:
    // Bias tiles are special in pto-isa; keep a safe fallback qualifier.
    return "__gm__";
  case pto::AddressSpace::SCALING:
    // pto-isa TileType::Scaling maps to __fbuf__ (see pto/common/memory.hpp).
    return "__fbuf__";
  }
  return "__gm__";
}

[[maybe_unused]] static constexpr llvm::StringLiteral kLoweredSetValidShapeAttrName =
    "__pto.lowered_set_validshape";
[[maybe_unused]] static constexpr llvm::StringLiteral kLoweredSetValidShapeConfigAttrName =
    "__pto.lowered_set_validshape_config";
[[maybe_unused]] static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";
[[maybe_unused]] static constexpr llvm::StringLiteral kGlobalTensorStridesAttrName =
    "__pto.globaltensor_strides";

Value mlir::pto::peelUnrealized(Value v) {
  if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
    return castOp.getOperand(0);
  return v;
}


static Value maybeWrapGlobalMemrefAsGlobalTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value loweredValue,
    Type originalType, Operation *anchor);

static bool hasCompatibleKnownExtentForMGather(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic ||
         lhs == rhs;
}

static bool isKnownUnitExtentForMGather(int64_t value) {
  return value == ShapedType::kDynamic || value == 1;
}

struct GatherScatterShapeLayoutInfo {
  SmallVec2<int64_t> shape;
  bool rowMajor = false;
  bool colMajor = false;
};

static std::optional<GatherScatterShapeLayoutInfo>
getGatherScatterShapeLayoutInfo(Type ty) {
  if (auto tileTy = dyn_cast<pto::TileBufType>(ty)) {
    ArrayRef<int64_t> validShape = tileTy.getValidShape();
    if (validShape.size() != kTileRank2D)
      return std::nullopt;

    GatherScatterShapeLayoutInfo info;
    info.shape.assign(validShape.begin(), validShape.end());
    int32_t blayout = tileTy.getBLayoutValueI32();
    info.rowMajor = blayout == static_cast<int32_t>(pto::BLayout::RowMajor);
    info.colMajor = blayout == static_cast<int32_t>(pto::BLayout::ColMajor);
    return info;
  }

  auto memRefTy = dyn_cast<MemRefType>(ty);
  if (!memRefTy || memRefTy.getRank() != static_cast<int64_t>(kTileRank2D))
    return std::nullopt;

  SmallVec4<int64_t> strides;
  int64_t offset = ShapedType::kDynamic;
  if (failed(getStridesAndOffset(memRefTy, strides, offset)) ||
      strides.size() != kTileRank2D)
    return std::nullopt;

  GatherScatterShapeLayoutInfo info;
  info.shape.assign(memRefTy.getShape().begin(), memRefTy.getShape().end());
  info.rowMajor = strides[1] == 1;
  info.colMajor = strides[0] == 1;
  return info;
}

static bool isRowCoalescedMGatherIndexType(Type dataTy, Type idxTy) {
  auto dataInfo = getGatherScatterShapeLayoutInfo(dataTy);
  auto idxInfo = getGatherScatterShapeLayoutInfo(idxTy);
  if (!dataInfo || !idxInfo)
    return false;

  const bool rowCoalesce1xR =
      idxInfo->rowMajor && isKnownUnitExtentForMGather(idxInfo->shape[0]) &&
      hasCompatibleKnownExtentForMGather(idxInfo->shape[1], dataInfo->shape[0]);
  const bool rowCoalesceRx1 =
      idxInfo->colMajor &&
      hasCompatibleKnownExtentForMGather(idxInfo->shape[0], dataInfo->shape[0]) &&
      isKnownUnitExtentForMGather(idxInfo->shape[1]);
  return rowCoalesce1xR || rowCoalesceRx1;
}

static std::optional<mlir::pto::Layout> getLayoutAttrFromOp(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<mlir::pto::LayoutAttr>("layout"))
    return attr.getLayout();
  return std::nullopt;
}

static std::optional<mlir::pto::Layout> resolveLayoutFromValueChain(Value v) {
  v = peelUnrealized(v);
  while (Operation *def = v.getDefiningOp()) {
    if (auto layout = getLayoutAttrFromOp(def))
      return layout;
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      v = peelUnrealized(subview.getSource());
      continue;
    }
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
      v = peelUnrealized(reinterpret.getSource());
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      v = peelUnrealized(cast.getSource());
      continue;
    }
    if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (unrealized->getNumOperands() == 0)
        break;
      v = peelUnrealized(unrealized.getOperand(0));
      continue;
    }
    break;
  }
  return std::nullopt;
}

std::optional<mlir::pto::Layout>
mlir::pto::resolveLayoutForGlobalTensor(Operation *anchor, Value basePtr) {
  if (auto layout = getLayoutAttrFromOp(anchor))
    return layout;
  return resolveLayoutFromValueChain(basePtr);
}

std::string mlir::pto::layoutToEmitCString(mlir::pto::Layout layout) {
  switch (layout) {
  case mlir::pto::Layout::ND:
    return "pto::Layout::ND";
  case mlir::pto::Layout::DN:
    return "pto::Layout::DN";
  case mlir::pto::Layout::NZ:
    return "pto::Layout::NZ";
  }
  return "pto::Layout::ND";
}

bool mlir::pto::isEmitCGlobalTensorLikeType(Type ty) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty);
  return opaqueTy && opaqueTy.getValue().contains("GlobalTensor<");
}

static std::optional<std::string> getEmitCFloat8TypeToken(Type elemTy) {
  if (pto::isPTOFloat8Type(elemTy) &&
      (elemTy.isFloat8E4M3() || elemTy.isFloat8E4M3FN() ||
       elemTy.isFloat8E4M3FNUZ() || elemTy.isFloat8E4M3B11FNUZ())) {
    return "float8_e4m3_t";
  }
  if (pto::isPTOFloat8Type(elemTy) &&
      (elemTy.isFloat8E5M2() || elemTy.isFloat8E5M2FNUZ())) {
    return "float8_e5m2_t";
  }
  if (isa<pto::HiF8Type>(elemTy))
    return "hifloat8_t";
  if (isa<pto::F4E1M2x2Type>(elemTy))
    return "float4_e1m2x2_t";
  if (isa<pto::F4E2M1x2Type>(elemTy))
    return "float4_e2m1x2_t";
  return std::nullopt;
}

static std::optional<std::string> getEmitCFloatingTypeToken(Type elemTy) {
  if (elemTy.isF16())
    return "half";
  if (elemTy.isBF16())
    return "bfloat16_t";
  if (elemTy.isF32())
    return "float";
  if (elemTy.isF64())
    return "double";
  return std::nullopt;
}

static std::optional<std::string> getEmitCIntegerTypeToken(Type elemTy) {
  auto intTy = dyn_cast<IntegerType>(elemTy);
  if (!intTy)
    return std::nullopt;
  const bool isSigned = intTy.isSignless() || intTy.isSigned();
  switch (intTy.getWidth()) {
  case kPTOI8BitWidth:
    return isSigned ? "int8_t" : "uint8_t";
  case kPTOI16BitWidth:
    return isSigned ? "int16_t" : "uint16_t";
  case kPTOI32BitWidth:
    return isSigned ? "int32_t" : "uint32_t";
  case kPTOI64BitWidth:
    return intTy.isUnsigned() ? "uint64_t" : "int64_t";
  default:
    return std::nullopt;
  }
}

std::string mlir::pto::getEmitCScalarTypeToken(Type elemTy) {
  if (auto tok = getEmitCFloat8TypeToken(elemTy))
    return *tok;
  if (auto tok = getEmitCFloatingTypeToken(elemTy))
    return *tok;
  if (auto tok = getEmitCIntegerTypeToken(elemTy))
    return *tok;
  return "float";
}

static emitc::PointerType getEmitCPointerType(MLIRContext *ctx,
                                              StringRef pointeeTypeStr) {
  return emitc::PointerType::get(emitc::OpaqueType::get(ctx, pointeeTypeStr));
}

static emitc::PointerType getEmitCPointerType(MLIRContext *ctx,
                                              StringRef qualifier,
                                              StringRef elemTypeStr) {
  return getEmitCPointerType(ctx, (qualifier + " " + elemTypeStr).str());
}

static bool isEmitCPointerLikeType(Type ty) {
  if (isa<emitc::PointerType>(ty))
    return true;
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty))
    return opaqueTy.getValue().ends_with("*");
  return false;
}

int64_t mlir::pto::getEmitCScalarByteWidth(Type elemTy) {
  if (pto::getPTOStorageElemByteSize(elemTy) == kPTOByteSize)
    return kPTOByteSize;
  if (elemTy.isF16() || elemTy.isBF16() || elemTy.isInteger(kPTOI16BitWidth))
    return kPTOHalfWordBytes;
  if (elemTy.isF32() || elemTy.isInteger(kPTOI32BitWidth))
    return kPTOWordBytes;
  if (elemTy.isF64() || elemTy.isInteger(kPTOI64BitWidth))
    return kPTODoubleWordBytes;
  return kPTOWordBytes;
}

Value mlir::pto::peelEmitCCasts(Value v) {
  v = peelUnrealized(v);
  if (auto castOp = v.getDefiningOp<emitc::CastOp>())
    v = castOp.getOperand();
  return v;
}

bool mlir::pto::isEmitCTileLikeValue(Value v) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(v.getType());
  if (!opaqueTy)
    return false;
  StringRef typeStr = opaqueTy.getValue();
  return typeStr.contains("Tile<") || typeStr.contains("ConvTile<");
}

Value mlir::pto::scalePackedTileDynamicDim(ConversionPatternRewriter &rewriter,
                                           Location loc, Type elemTy,
                                           pto::BLayout blayout, Value emitted,
                                           int dimIdx) {
  if (!emitted || !pto::isPTOFloat4PackedType(elemTy))
    return emitted;
  int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
  if (dimIdx != packedDim)
    return emitted;
  auto i32Ty = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
  Value two = makeEmitCIntConstant(rewriter, loc, i32Ty, 2);
  return rewriter.create<emitc::MulOp>(loc, i32Ty, emitted, two).getResult();
}

Value mlir::pto::buildTileCtorDimValue(ConversionPatternRewriter &rewriter,
                                       Location loc, Value emitted,
                                       int64_t fallback) {
  if (emitted)
    return emitted;
  return makeEmitCIntConstant(rewriter, loc,
                              emitc::OpaqueType::get(rewriter.getContext(), "int32_t"),
                              fallback);
}

std::string mlir::pto::getTileRoleToken(Attribute memorySpace) {
  if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace)) {
    switch (asAttr.getAddressSpace()) {
    case pto::AddressSpace::VEC:
      return "TileType::Vec";
    case pto::AddressSpace::MAT:
      return "TileType::Mat";
    case pto::AddressSpace::LEFT:
      return "TileType::Left";
    case pto::AddressSpace::RIGHT:
      return "TileType::Right";
    case pto::AddressSpace::ACC:
      return "TileType::Acc";
    case pto::AddressSpace::BIAS:
      return "TileType::Bias";
    case pto::AddressSpace::SCALING:
      return "TileType::Scaling";
    case pto::AddressSpace::GM:
    case pto::AddressSpace::Zero:
      break;
    }
  }
  return "TileType::Vec";
}

std::string mlir::pto::getTileBufCompactToken(pto::TileBufConfigAttr configAttr) {
  std::string compactTok = "CompactMode::Null";
  if (auto compactAttr = dyn_cast<CompactModeAttr>(configAttr.getCompactMode())) {
    switch (static_cast<int32_t>(compactAttr.getValue())) {
    case 1:
      compactTok = "CompactMode::Normal";
      break;
    case kNumber2:
      compactTok = "CompactMode::RowPlusOne";
      break;
    default:
      break;
    }
  }
  return compactTok;
}

std::optional<std::string> mlir::pto::getEmitCTileTypeString(pto::TileBufType type) {
  if (type.getRank() != static_cast<int64_t>(kTileRank2D))
    return std::nullopt;
  auto validShape = type.getValidShape();
  if (validShape.size() != kTileRank2D)
    return std::nullopt;

  Type elemTy = type.getElementType();
  auto configAttr = type.getConfigAttr();
  pto::BLayout blayout = getTileBufBLayoutValue(configAttr);
  ArrayRef<int64_t> shape = type.getShape();
  int64_t rows = shape[0];
  int64_t cols = shape[1];

  auto render = [elemTy, blayout](int64_t dim, int dimIdx) {
    return renderTileTemplateDim(dim, elemTy, blayout, dimIdx);
  };

  std::string vrowTok =
      validShape[0] == ShapedType::kDynamic
          ? "-1"
          : std::to_string(render(validShape[0], 0));
  std::string vcolTok =
      validShape[1] == ShapedType::kDynamic
          ? "-1"
          : std::to_string(render(validShape[1], 1));

  int32_t fractal = kFractalSize512;
  if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
    fractal = frAttr.getInt();
  return std::string("Tile<") + getTileRoleToken(type.getMemorySpace()) + ", " +
         getEmitCScalarTypeToken(elemTy) + ", " +
         std::to_string(render(rows, 0)) + ", " +
         std::to_string(render(cols, 1)) + ", " +
         getTileBufBLayoutToken(configAttr) + ", " + vrowTok + ", " + vcolTok +
         ", " + getTileBufSLayoutToken(configAttr) + ", " +
         std::to_string(fractal) + ", " + getTileBufPadToken(configAttr) + ", " +
         getTileBufCompactToken(configAttr) + ">";
}

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class PTOToEmitCTypeConverter : public TypeConverter {
public:
  PTOToEmitCTypeConverter(MLIRContext *Ctx, PTOArch) {
    addScalarConversions(Ctx);
    addPTOScalarConversions(Ctx);
    addPointerAndArrayConversions(Ctx);
    addAsyncAndViewConversions(Ctx);
    addMemRefAndFunctionConversions(Ctx);
    addMaterializations();
  }

private:
  void addScalarConversions(MLIRContext *Ctx) {
    addConversion([Ctx](FloatType type) -> Type {
      if (auto tok = getEmitCFloat8TypeToken(type))
        return emitc::OpaqueType::get(Ctx, *tok);
      if (auto tok = getEmitCFloatingTypeToken(type))
        return emitc::OpaqueType::get(Ctx, *tok);
      llvm::errs() << "[Debug] Unsupported FloatType: " << type << "\n";
      return Type{};
    });

    addConversion([Ctx](IntegerType type) -> Type {
      if (type.getWidth() == 1)
        return type;
      if (auto tok = getEmitCIntegerTypeToken(type))
        return emitc::OpaqueType::get(Ctx, *tok);
      llvm::errs() << "[Debug] Unsupported IntegerType width: "
                   << type.getWidth() << "\n";
      return emitc::OpaqueType::get(Ctx, "int32_t");
    });

    addConversion([Ctx](IndexType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "int32_t");
    });

    addConversion([Ctx](VectorType type) -> Type {
      if (type.getRank() == kNumber1 && type.getNumElements() == kNumber4 &&
          type.getElementType().isInteger(kPTOI16BitWidth)) {
        return emitc::OpaqueType::get(Ctx, "pto::MrgSortExecutedNumList");
      }
      return Type{};
    });
  }

  void addPTOScalarConversions(MLIRContext *Ctx) {
    addConversion([Ctx](pto::HiF8Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "hifloat8_t");
    });
    addConversion([Ctx](pto::F4E1M2x2Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "float4_e1m2x2_t");
    });
    addConversion([Ctx](pto::F4E2M1x2Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "float4_e2m1x2_t");
    });
    addConversion([](emitc::OpaqueType type) { return type; });
    addConversion([](emitc::PointerType type) { return type; });
  }

  void addPointerAndArrayConversions(MLIRContext *Ctx) {
    addConversion([this, Ctx](pto::PtrType type) -> std::optional<Type> {
      Type newElemType = convertType(type.getElementType());
      auto opq = dyn_cast_or_null<emitc::OpaqueType>(newElemType);
      if (!opq) {
        llvm::errs() << "  [Error] PtrType elem type is not OpaqueType: "
                     << newElemType << "\n";
        return std::nullopt;
      }
      return getEmitCPointerType(Ctx, "__gm__ " + opq.getValue().str());
    });

    addConversion([Ctx](pto::PipeType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "auto");
    });
    addConversion([Ctx](pto::EventIdArrayType type) -> Type {
      return emitc::OpaqueType::get(
          Ctx, "PTOAS_EventIdArray<" + std::to_string(type.getSize()) + ">");
    });
    addConversion([this](pto::LocalArrayType type) -> std::optional<Type> {
      Type convertedElem = convertType(type.getElementType());
      if (!convertedElem)
        return std::nullopt;
      return emitc::ArrayType::get(type.getShape(), convertedElem);
    });
  }

  void addAsyncAndViewConversions(MLIRContext *Ctx) {
    addConversion([Ctx](pto::AsyncSessionType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "pto::comm::AsyncSession");
    });
    addConversion([Ctx](pto::AsyncEventType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "pto::comm::AsyncEvent");
    });
    addConversion([Ctx](pto::PrefetchAsyncContextType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "pto::PrefetchAsyncContext");
    });
    addConversion([Ctx](pto::TensorViewType type) -> Type {
      return getGlobalTensorOpaqueTypeFromShape(Ctx, type.getElementType(),
                                                type.getShape());
    });
    addConversion([Ctx](pto::PartitionTensorViewType type) -> Type {
      return getGlobalTensorOpaqueTypeFromShape(Ctx, type.getElementType(),
                                                type.getShape());
    });
    addConversion([Ctx](pto::TileBufType type) -> std::optional<Type> {
      auto typeString = getEmitCTileTypeString(type);
      if (!typeString)
        return std::nullopt;
      return emitc::OpaqueType::get(Ctx, *typeString);
    });
  }

  void addMemRefAndFunctionConversions(MLIRContext *Ctx) {
    addConversion([this, Ctx](MemRefType type) -> std::optional<Type> {
      LLVM_DEBUG(llvm::dbgs() << "Converting MemRef: " << type << "\n");
      Type newElemType = convertType(type.getElementType());
      auto opq = dyn_cast_or_null<emitc::OpaqueType>(newElemType);
      if (!opq) {
        llvm::errs() << "  [Error] Converted element type is not OpaqueType: "
                     << newElemType << "\n";
        return std::nullopt;
      }

      std::string qualifier = "__gm__";
      Attribute memorySpace = type.getMemorySpace();
      if (auto ptoAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace)) {
        qualifier = addrSpaceQualifier(ptoAttr.getAddressSpace());
      } else if (memorySpace) {
        llvm::errs() << "  [Warning] Unknown MemorySpace Attribute type: "
                     << memorySpace << "\n";
      }

      std::string finalTypeStr = qualifier + " " + opq.getValue().str();
      LLVM_DEBUG(llvm::dbgs() << "  [Success] -> " << finalTypeStr << "*\n");
      return getEmitCPointerType(Ctx, finalTypeStr);
    });

    addConversion([this](FunctionType type) -> Type {
      SmallVector<Type> inputs;
      SmallVector<Type> results;
      if (failed(convertTypes(type.getInputs(), inputs)) ||
          failed(convertTypes(type.getResults(), results))) {
        return Type{};
      }
      return FunctionType::get(type.getContext(), inputs, results);
    });
  }

  void addMaterializations() {
    auto materializeCast = [](OpBuilder &Builder, Type ResultType,
                              ValueRange Inputs, Location Loc) -> Value {
      if (Inputs.size() != 1)
        return Value();
      return Builder
          .create<UnrealizedConversionCastOp>(Loc, ResultType, Inputs[0])
          .getResult(0);
    };
    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
    addArgumentMaterialization(materializeCast);
  }
};

[[maybe_unused]] static constexpr unsigned kPTOIndexBitWidth =
    32; // keep consistent with IndexType conversion

bool mlir::pto::isSetFFTsPointerLikeType(Type ty) {
  return isEmitCPointerLikeType(ty);
}

bool mlir::pto::tileDataReturnsIntegralAddress(pto::AddressSpace as) {
  return as == pto::AddressSpace::BIAS;
}

static Type getTileDataResultType(MLIRContext *ctx, pto::AddressSpace as,
                                  StringRef elemTok) {
  if (tileDataReturnsIntegralAddress(as))
    return emitc::OpaqueType::get(ctx, "uint64_t");
  return getEmitCPointerType(ctx, addrSpaceQualifier(as), elemTok);
}

Value mlir::pto::materializeTileDataValue(ConversionPatternRewriter &rewriter,
                                          Location loc, Value tile,
                                          pto::AddressSpace as,
                                          StringRef elemTypeToken) {
  auto rawTy =
      getTileDataResultType(rewriter.getContext(), as, elemTypeToken);
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, rawTy, "PTOAS__TILE_DATA",
                                   ArrayAttr{}, ArrayAttr{},
                                   ValueRange{tile})
      .getResult(0);
}

Value mlir::pto::materializeAddressAsPointer(ConversionPatternRewriter &rewriter,
                                         Location loc, Value addr,
                                         pto::AddressSpace as,
                                         StringRef elemTok) {
  auto *ctx = rewriter.getContext();
  std::string ptrTyStr =
      std::string(addrSpaceQualifier(as)) + " " + elemTok.str() + "*";
  auto ptrTy = getEmitCPointerType(ctx, addrSpaceQualifier(as), elemTok);
  if (isSetFFTsPointerLikeType(addr.getType())) {
    if (addr.getType() == ptrTy)
      return addr;
    return rewriter.create<emitc::CastOp>(loc, ptrTy, addr).getResult();
  }
  auto castTyAttr =
      rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, ptrTyStr)});
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, ptrTy, "reinterpret_cast",
                                   ArrayAttr{}, castTyAttr,
                                   ValueRange{addr})
      .getResult(0);
}

static bool hasInterCoreSyncOp(func::FuncOp func) {
  bool found = false;
  func.walk([&found](Operation *op) {
    if (isa<pto::SyncSetOp, pto::SyncWaitOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool hasSetFFTsOp(func::FuncOp func) {
  bool found = false;
  func.walk([&found](Operation *op) {
    if (isa<pto::SetFFTsOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

//===----------------------------------------------------------------------===//
// EmitC scalar helpers
//===----------------------------------------------------------------------===//
Value mlir::pto::makeEmitCOpaqueConstant(ConversionPatternRewriter &rewriter,
                                     Location loc, Type type,
                                     llvm::StringRef literal) {
  auto attr = emitc::OpaqueAttr::get(rewriter.getContext(), literal);
  return rewriter.create<emitc::ConstantOp>(loc, type, attr);
}

Value mlir::pto::makeEmitCIntConstant(ConversionPatternRewriter &rewriter,
                                  Location loc, Type type, int64_t value) {
  return makeEmitCOpaqueConstant(rewriter, loc, type, std::to_string(value));
}
Value mlir::pto::emitCCast(ConversionPatternRewriter &rewriter, Location loc,
                       Type dstType, Value src) {
  if (src.getType() == dstType)
    return src;
  return rewriter.createOrFold<emitc::CastOp>(loc, dstType, src);
}

// For signless iN integers lowered to signed C++ types, this creates a value
// representing the same N-bit pattern in an unsigned C++ type of the same
// width. This avoids incorrect sign-extension when later widening to a larger
// unsigned type.
Value mlir::pto::castSignlessIntToUnsignedSameWidth(ConversionPatternRewriter &rewriter,
                                                Location loc, Value v,
                                                unsigned bitWidth) {
  auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
  return emitCCast(rewriter, loc, uTy, v);
}

//===----------------------------------------------------------------------===//
// pto.mgather lowering -> MGATHER(dst, src, indexes)  (pto-isa)
//===----------------------------------------------------------------------===//

struct PTOMGatherToMGATHER : public OpConversionPattern<pto::MGatherOp> {
  using OpConversionPattern<pto::MGatherOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MGatherOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Value mem = peelUnrealized(adaptor.getMem());
    Value idx = peelUnrealized(adaptor.getIdx());
    Value dst = peelUnrealized(adaptor.getDst());

    Value memArg = maybeWrapGlobalMemrefAsGlobalTensor(
        rewriter, op.getLoc(), mem, op.getMem().getType(), op.getOperation());

    auto gatherOobTok = [](pto::GatherOOB mode) -> StringRef {
      switch (mode) {
      case pto::GatherOOB::Undefined:
        return "pto::GatherOOB::Undefined";
      case pto::GatherOOB::Clamp:
        return "pto::GatherOOB::Clamp";
      case pto::GatherOOB::Wrap:
        return "pto::GatherOOB::Wrap";
      case pto::GatherOOB::Zero:
        return "pto::GatherOOB::Zero";
      }
      llvm_unreachable("unknown GatherOOB");
    };

    SmallVec2<Attribute> templateArgVec;
    const bool rowCoalesce =
        isRowCoalescedMGatherIndexType(op.getDst().getType(), op.getIdx().getType());
    templateArgVec.push_back(emitc::OpaqueAttr::get(
        ctx, rowCoalesce ? "pto::Coalesce::Row" : "pto::Coalesce::Elem"));
    if (op.getGatherOob() != pto::GatherOOB::Undefined) {
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, gatherOobTok(op.getGatherOob())));
    }
    ArrayAttr templateArgs = rewriter.getArrayAttr(templateArgVec);

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "MGATHER",
        ArrayAttr{}, templateArgs,
        ValueRange{dst, memArg, idx});
    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, dst);
    }
    return success();
  }
};

static std::optional<StringRef> getKernelKindMacro(func::FuncOp funcOp) {
  auto kernelKindAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(FunctionKernelKindAttr::name);
  if (!kernelKindAttr)
    return std::nullopt;

  switch (kernelKindAttr.getKernelKind()) {
  case FunctionKernelKind::Cube:
    return StringRef("__DAV_CUBE__");
  case FunctionKernelKind::Vector:
    return StringRef("__DAV_VEC__");
  }

  llvm_unreachable("unexpected kernel kind");
}

struct FuncToEmitC : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  static void copyFunctionAttrs(func::FuncOp op, emitc::FuncOp emitcFunc) {
    for (const auto &namedAttr : op->getAttrs()) {
      StringRef name = namedAttr.getName().strref();
      if (name == op.getFunctionTypeAttrName() ||
          name == SymbolTable::getSymbolAttrName() ||
          name == pto::kPTOEntryAttrName ||
          name == pto::kLegacyHACCEntryAttrName ||
          name == "pto.internal.entry") {
        continue;
      }
      emitcFunc->setAttr(namedAttr.getName(), namedAttr.getValue());
    }
  }

  static void setFunctionSpecifiers(func::FuncOp op, emitc::FuncOp emitcFunc,
                                    ConversionPatternRewriter &rewriter) {
    if (pto::isPTOEntryFunction(op)) {
      emitcFunc.setSpecifiersAttr(
          rewriter.getStrArrayAttr({"__global__ AICORE"}));
    } else if (op.isPrivate()) {
      emitcFunc.setSpecifiersAttr(
          rewriter.getStrArrayAttr({"static", "AICORE"}));
    } else {
      emitcFunc.setSpecifiersAttr(rewriter.getStrArrayAttr({"AICORE"}));
    }
  }

  static LogicalResult inlineAndConvertFuncBody(
      func::FuncOp op, emitc::FuncOp emitcFunc, FunctionType funcType,
      const TypeConverter &typeConverter,
      ConversionPatternRewriter &rewriter) {
    rewriter.inlineRegionBefore(op.getBody(), emitcFunc.getBody(), emitcFunc.end());

    TypeConverter::SignatureConversion entryConv(op.getNumArguments());
    for (unsigned i = 0; i < op.getNumArguments(); ++i)
      entryConv.addInputs(i, funcType.getInput(i));
    return rewriter.convertRegionTypes(&emitcFunc.getBody(), typeConverter,
                                       &entryConv);
  }

  static void emitFunctionPrologue(func::FuncOp op, emitc::FuncOp emitcFunc,
                                   std::optional<StringRef> kernelKindMacro,
                                   bool needsNoSplitGuard,
                                   ConversionPatternRewriter &rewriter) {
    Block &entryBlock = emitcFunc.getBody().front();
    rewriter.setInsertionPointToStart(&entryBlock);
    rewriter.create<emitc::VerbatimOp>(op.getLoc(), "using T = float;");
    if (!kernelKindMacro)
      return;
    std::string startMacro = "\n#if defined(" + kernelKindMacro->str() + ")";
    rewriter.create<emitc::VerbatimOp>(op.getLoc(), startMacro);
    if (*kernelKindMacro != "__DAV_VEC__")
      return;
    rewriter.create<emitc::VerbatimOp>(op.getLoc(), "set_mask_norm();");
    rewriter.create<emitc::VerbatimOp>(op.getLoc(), "set_vector_mask(-1, -1);");
    if (needsNoSplitGuard)
      rewriter.create<emitc::VerbatimOp>(op.getLoc(),
                                         "if (get_subblockid() == 0) {");
  }

  static void emitFunctionEpilogue(func::FuncOp op, emitc::FuncOp emitcFunc,
                                   std::optional<StringRef> kernelKindMacro,
                                   bool needsNoSplitGuard,
                                   ConversionPatternRewriter &rewriter) {
    if (!kernelKindMacro)
      return;
    Block &lastBlock = emitcFunc.getBody().back();
    rewriter.setInsertionPoint(lastBlock.getTerminator());
    if (*kernelKindMacro == "__DAV_VEC__" && needsNoSplitGuard)
      rewriter.create<emitc::VerbatimOp>(op.getLoc(), "}");
    std::string endMacro = "#endif // " + kernelKindMacro->str() + "\n";
    rewriter.create<emitc::VerbatimOp>(op.getLoc(), endMacro);
  }

  LogicalResult matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type convertedTy = getTypeConverter()->convertType(op.getFunctionType());
    auto funcType = dyn_cast_or_null<FunctionType>(convertedTy);
    if (!funcType)
      return rewriter.notifyMatchFailure(op, "failed to convert function type");
    if (funcType.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "EmitC cannot return multiple values");

    auto emitcFunc =
        rewriter.create<emitc::FuncOp>(op.getLoc(), op.getName(), funcType);
    copyFunctionAttrs(op, emitcFunc);
    if (op.isDeclaration()) {
      emitcFunc.setSpecifiersAttr(rewriter.getStrArrayAttr({"extern"}));
      rewriter.eraseOp(op);
      return success();
    }

    setFunctionSpecifiers(op, emitcFunc, rewriter);
    std::optional<StringRef> kernelKindMacro = getKernelKindMacro(op);
    bool needsNoSplitGuard = needsA5NoSplitVectorGuard(op.getOperation());
    if (failed(inlineAndConvertFuncBody(op, emitcFunc, funcType,
                                        *getTypeConverter(), rewriter))) {
      return failure();
    }
    emitFunctionPrologue(op, emitcFunc, kernelKindMacro, needsNoSplitGuard,
                         rewriter);
    emitFunctionEpilogue(op, emitcFunc, kernelKindMacro, needsNoSplitGuard,
                         rewriter);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper: build GlobalTensor from a static MemRef (for TLOAD/TSTORE)
//===----------------------------------------------------------------------===//

std::string mlir::pto::getElemTypeStringForGT(Type elemTy) {
  return getEmitCScalarTypeToken(elemTy);
}

static bool hasStaticShape(MemRefType mrTy) {
  return llvm::none_of(mrTy.getShape(), [](int64_t dim) {
    return dim == ShapedType::kDynamic;
  });
}

static bool getStaticMemrefLayout(MemRefType mrTy, SmallVectorImpl<int64_t> &strides,
                                  int64_t &offset) {
  if (failed(getStridesAndOffset(mrTy, strides, offset))) {
    strides.clear();
    int64_t stride = 1;
    ArrayRef<int64_t> shape = mrTy.getShape();
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      strides.push_back(stride);
      stride *= shape[i];
    }
    std::reverse(strides.begin(), strides.end());
    offset = 0;
  }
  return offset != ShapedType::kDynamic &&
         llvm::none_of(strides, [](int64_t strideValue) {
           return strideValue == ShapedType::kDynamic;
         });
}

Value mlir::pto::applyStaticMemrefOffset(ConversionPatternRewriter &rewriter,
                                     Location loc, Value basePtr,
                                     int64_t offset) {
  if (offset == 0)
    return basePtr;
  auto *ctx = rewriter.getContext();
  Type u32Ty = emitc::OpaqueType::get(ctx, "unsigned");
  auto offVal = rewriter.create<emitc::ConstantOp>(
      loc, u32Ty, emitc::OpaqueAttr::get(ctx, std::to_string(offset)));
  return rewriter.create<emitc::AddOp>(loc, basePtr.getType(), basePtr, offVal);
}

static int getGlobalTensorElementBytes(Type elemTy) {
  return static_cast<int>(getPTOStorageElemByteSize(elemTy));
}

int64_t mlir::pto::multiplyOrDynamic(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0)
    return -1;
  return lhs * rhs;
}

void mlir::pto::buildGlobalTensorShapeAndStride(ArrayRef<int64_t> shape,
                                            ArrayRef<int64_t> strides,
                                            SmallVectorImpl<int64_t> &shape5D,
                                            SmallVectorImpl<int64_t> &stride5D) {
  shape5D.assign(kPTOPaddedTensorRank5D, 1);
  stride5D.assign(kPTOPaddedTensorRank5D, 1);
  int rank = static_cast<int>(shape.size());
  int shift = static_cast<int>(kPTOPaddedTensorRank5D) - rank;
  for (int i = 0; i < rank &&
                  i < static_cast<int>(kPTOPaddedTensorRank5D);
       ++i) {
    shape5D[shift + i] = shape[i];
    stride5D[shift + i] = strides[i];
  }
  for (int i = 3; i >= 0; --i) {
    if (i >= shift)
      continue;
    stride5D[i] = multiplyOrDynamic(shape5D[i + 1], stride5D[i + 1]);
  }
}

std::string mlir::pto::joinIntTemplateParams(ArrayRef<int64_t> values) {
  std::string result;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      result += ", ";
    result += std::to_string(values[i]);
  }
  return result;
}

SmallVector<int64_t> mlir::pto::buildRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  int64_t running = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    strides[i] = running;
    running = multiplyOrDynamic(running, shape[i]);
  }
  return strides;
}

static std::string getGlobalTensorTypeStringFromShape(Type elemTy,
                                                      ArrayRef<int64_t> shape,
                                                      StringRef layoutEnum) {
  SmallVector<int64_t> strides = buildRowMajorStrides(shape);
  return getGlobalTensorTypeStringFromShapeAndStrides(elemTy, shape, strides,
                                                      layoutEnum);
}

std::string mlir::pto::getGlobalTensorTypeStringFromShapeAndStrides(
    Type elemTy, ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
    StringRef layoutEnum) {
  SmallVec5<int64_t> shape5D;
  SmallVec5<int64_t> stride5D;
  buildGlobalTensorShapeAndStride(shape, strides, shape5D, stride5D);

  std::string elemTypeStr = getElemTypeStringForGT(elemTy);
  std::string shapeType = "pto::Shape<" + joinIntTemplateParams(shape5D) + ">";
  std::string strideType =
      "pto::Stride<" + joinIntTemplateParams(stride5D) + ">";
  return "GlobalTensor<" + elemTypeStr + ", " + shapeType + ", " +
         strideType + ", " + layoutEnum.str() + ">";
}

static emitc::OpaqueType getGlobalTensorOpaqueTypeFromShape(
    MLIRContext *ctx, Type elemTy, ArrayRef<int64_t> shape,
    StringRef layoutEnum) {
  return emitc::OpaqueType::get(
      ctx, getGlobalTensorTypeStringFromShape(elemTy, shape, layoutEnum));
}

static std::string inferFallbackGlobalTensorLayout(ArrayRef<int64_t> shape5D,
                                                   ArrayRef<int64_t> stride5D,
                                                   Type elemTy) {
  int elemBytes = getGlobalTensorElementBytes(elemTy);
  if (elemBytes == 0)
    return "pto::Layout::ND";
  if (shape5D[kPaddedShapeInnerRowDim2] == kFractalSize16 &&
      multiplyOrDynamic(shape5D[kPaddedShapeInnerRowDim2],
                        shape5D[kPaddedShapeInnerColDim3]) *
              elemBytes ==
          kFractalSize512 &&
      stride5D[kPaddedShapeInnermostDim4] == kPaddedShapeUnitStride1 &&
      stride5D[kPaddedShapeInnerColDim3] ==
          shape5D[kPaddedShapeInnermostDim4]) {
    return "pto::Layout::NZ";
  }

  bool isRowMajor = stride5D[kPaddedShapeInnermostDim4] == kPaddedShapeUnitStride1;
  for (int i = static_cast<int>(kPaddedShapeInnerColDim3); i >= 0 && isRowMajor; --i)
    isRowMajor = stride5D[i] == multiplyOrDynamic(stride5D[i + 1], shape5D[i + 1]);

  bool isColMajor = stride5D[0] == kPaddedShapeUnitStride1;
  for (int i = 0; i < static_cast<int>(kNumber4) && isColMajor; ++i)
    isColMajor = stride5D[i + 1] == multiplyOrDynamic(stride5D[i], shape5D[i]);
  if (isColMajor)
    return "pto::Layout::DN";
  return isRowMajor ? "pto::Layout::ND" : "pto::Layout::ND";
}

static std::string resolveGlobalTensorLayout(Operation *anchor, Value basePtr,
                                             ArrayRef<int64_t> shape5D,
                                             ArrayRef<int64_t> stride5D,
                                             Type elemTy) {
  if (auto layout = resolveLayoutForGlobalTensor(anchor, basePtr))
    return layoutToEmitCString(*layout);
  return inferFallbackGlobalTensorLayout(shape5D, stride5D, elemTy);
}

struct GlobalTensorTypeNames {
  std::string shapeTypeName;
  std::string strideTypeName;
  std::string tensorTypeName;
  std::string layoutConstName;
};

static GlobalTensorTypeNames getGlobalTensorTypeNames(Operation *anchor) {
  std::string suffix =
      "_" + std::to_string(
                static_cast<size_t>(llvm::hash_value(
                    static_cast<const void *>(anchor))));
  return {
      "GTShape" + suffix,
      "GTStride" + suffix,
      "GT" + suffix,
      "GT" + suffix + "_layout",
  };
}

static void emitGlobalTensorTypeAliases(ConversionPatternRewriter &rewriter,
                                        Location loc,
                                        const GlobalTensorTypeNames &names,
                                        ArrayRef<int64_t> shape5D,
                                        ArrayRef<int64_t> stride5D,
                                        StringRef elemTypeStr,
                                        StringRef layoutEnum) {
  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + names.shapeTypeName + " = pto::Shape<" +
               joinIntTemplateParams(shape5D) + ">;");
  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + names.strideTypeName + " = pto::Stride<" +
               joinIntTemplateParams(stride5D) + ">;");
  rewriter.create<emitc::VerbatimOp>(loc, "constexpr pto::Layout " +
                                              names.layoutConstName + " = " +
                                              layoutEnum.str() + ";");
  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + names.tensorTypeName + " = GlobalTensor<" +
               elemTypeStr.str() + ", " + names.shapeTypeName + ", " +
               names.strideTypeName + ", " + names.layoutConstName + ">;");
}

static SmallVector<Value> buildGlobalTensorCtorArgs(
    ConversionPatternRewriter &rewriter, Location loc,
    const GlobalTensorTypeNames &names, Value ptr) {
  auto *ctx = rewriter.getContext();
  auto shapeTypeOpaque = emitc::OpaqueType::get(ctx, names.shapeTypeName);
  auto strideTypeOpaque = emitc::OpaqueType::get(ctx, names.strideTypeName);
  Value shapeValue =
      rewriter
          .create<emitc::CallOpaqueOp>(loc, shapeTypeOpaque, names.shapeTypeName,
                                       ArrayAttr{}, ArrayAttr{}, ValueRange{})
          .getResult(0);
  Value strideValue =
      rewriter
          .create<emitc::CallOpaqueOp>(loc, strideTypeOpaque, names.strideTypeName,
                                       ArrayAttr{}, ArrayAttr{}, ValueRange{})
          .getResult(0);
  return {ptr, shapeValue, strideValue};
}

Value mlir::pto::buildGlobalTensorFromMemref(ConversionPatternRewriter &rewriter,
                                         Location loc, Value basePtr,
                                         MemRefType mrTy,
                                         Operation *anchor) {
  auto *ctx = rewriter.getContext();

  ArrayRef<int64_t> shape = mrTy.getShape();
  if (!hasStaticShape(mrTy))
    return Value();

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (!getStaticMemrefLayout(mrTy, strides, offset))
    return Value();

  Value ptr = applyStaticMemrefOffset(rewriter, loc, basePtr, offset);
  GlobalTensorTypeNames names = getGlobalTensorTypeNames(anchor);
  std::string elemTypeStr = getElemTypeStringForGT(mrTy.getElementType());
  SmallVec5<int64_t> shape5D;
  SmallVec5<int64_t> stride5D;
  buildGlobalTensorShapeAndStride(shape, strides, shape5D, stride5D);

  std::string layoutEnum = resolveGlobalTensorLayout(
      anchor, basePtr, shape5D, stride5D, mrTy.getElementType());
  emitGlobalTensorTypeAliases(rewriter, loc, names, shape5D, stride5D,
                              elemTypeStr, layoutEnum);
  auto gtType = emitc::OpaqueType::get(ctx, names.tensorTypeName);
  SmallVector<Value> gtArgs = buildGlobalTensorCtorArgs(rewriter, loc, names, ptr);

  auto gtInst = rewriter.create<emitc::CallOpaqueOp>(
      loc, gtType, names.tensorTypeName, ArrayAttr{}, ArrayAttr{},
      ValueRange(gtArgs));
  return gtInst.getResult(0);
}

static Value maybeWrapGlobalMemrefAsGlobalTensor(
    ConversionPatternRewriter &rewriter, Location loc, Value loweredValue,
    Type originalType, Operation *anchor) {
  auto mrTy = dyn_cast<MemRefType>(originalType);
  if (!mrTy)
    return loweredValue;

  bool isGlobal = true;
  if (auto asAttr =
          dyn_cast_or_null<pto::AddressSpaceAttr>(mrTy.getMemorySpace())) {
    auto as = asAttr.getAddressSpace();
    isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
  }
  if (!isGlobal)
    return loweredValue;
  if (Value gt =
          buildGlobalTensorFromMemref(rewriter, loc, loweredValue, mrTy, anchor))
    return gt;
  return loweredValue;
}

Value mlir::pto::castToGMBytePointer(ConversionPatternRewriter &rewriter,
                                 Location loc, Value value) {
  auto *ctx = rewriter.getContext();
  auto targetTy =
      emitc::PointerType::get(emitc::OpaqueType::get(ctx, "__gm__ uint8_t"));
  if (value.getType() == targetTy)
    return value;

  auto castTyAttr =
      rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "__gm__ uint8_t*")});
  if (isSetFFTsPointerLikeType(value.getType())) {
    return rewriter
        .create<emitc::CallOpaqueOp>(loc, targetTy, "reinterpret_cast",
                                     ArrayAttr{}, castTyAttr,
                                     ValueRange{value})
        .getResult(0);
  }
  return rewriter.create<emitc::CastOp>(loc, targetTy, value).getResult();
}

Value mlir::pto::materializeTensorViewDataPointer(
    ConversionPatternRewriter &rewriter, Location loc, Value value,
    Type originalType) {
  auto tvTy = dyn_cast<pto::TensorViewType>(originalType);
  if (!tvTy)
    return value;

  auto *ctx = rewriter.getContext();
  std::string elemTypeStr = getElemTypeStringForGT(tvTy.getElementType());
  auto ptrTy = emitc::PointerType::get(
      emitc::OpaqueType::get(ctx, "__gm__ " + elemTypeStr));
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, ptrTy, "PTOAS__GLOBAL_TENSOR_DATA",
                                   ArrayAttr{}, ArrayAttr{}, ValueRange{value})
      .getResult(0);
}
static pto::TileBufConfigAttr getScratchTileConfigAttr(Value originalScratch,
                                                       MLIRContext *ctx) {
  pto::TileBufConfigAttr configAttr = pto::TileBufConfigAttr::getDefault(ctx);
  if (auto bind = originalScratch.getDefiningOp<pto::BindTileOp>())
    return bind.getConfig();
  if (auto cast = originalScratch.getDefiningOp<pto::PointerCastOp>()) {
    if (auto config = cast.getConfig())
      return *config;
  }
  return configAttr;
}

static std::string buildScratchTileTypeString(Type elemTy, int64_t rows,
                                              int64_t cols,
                                              pto::TileBufConfigAttr configAttr) {
  int32_t fractal = kFractalSize512;
  if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
    fractal = frAttr.getInt();
  pto::BLayout blayout = getTileBufBLayoutValue(configAttr);
  int64_t templateRows = renderTileTemplateDim(rows, elemTy, blayout, 0);
  int64_t templateCols = renderTileTemplateDim(cols, elemTy, blayout, 1);
  std::string elemTypeStr = getEmitCScalarTypeToken(elemTy);
  return "Tile<TileType::Vec, " + elemTypeStr + ", " +
         std::to_string(templateRows) + ", " +
         std::to_string(templateCols) + ", " +
         getTileBufBLayoutToken(configAttr) + ", " +
         std::to_string(templateRows) +
         ", " + std::to_string(templateCols) + ", " +
         getTileBufSLayoutToken(configAttr) + ", " +
         std::to_string(fractal) + ", " + getTileBufPadToken(configAttr) + ">";
}

Value mlir::pto::castAddressToU64(ConversionPatternRewriter &rewriter,
                                  Location loc, Value value) {
  auto *ctx = rewriter.getContext();
  auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
  if (isSetFFTsPointerLikeType(value.getType())) {
    auto addrTyAttr =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
    return rewriter
        .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast", ArrayAttr{},
                                     addrTyAttr, ValueRange{value})
        .getResult(0);
  }
  if (value.getType() == u64Ty)
    return value;
  return rewriter.create<emitc::CastOp>(loc, u64Ty, value).getResult();
}

FailureOr<Value> mlir::pto::buildAsyncScratchTileValue(
    ConversionPatternRewriter &rewriter, Location loc, Value originalScratch,
    Value emittedScratch) {
  Value scratch = peelUnrealized(emittedScratch);
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(scratch.getType())) {
    StringRef typeStr = opaqueTy.getValue();
    if (typeStr.contains("Tile<") || typeStr.contains("ConvTile<"))
      return scratch;
  }

  auto memTy = dyn_cast<MemRefType>(originalScratch.getType());
  if (!memTy)
    return failure();

  ArrayRef<int64_t> shape = memTy.getShape();
  if (!memTy.hasStaticShape() || shape.empty() || shape.size() > kTileRank2D)
    return failure();

  int64_t rows = shape.size() == 1 ? 1 : shape[0];
  int64_t cols = shape.size() == 1 ? shape[0] : shape[1];

  auto *ctx = rewriter.getContext();
  pto::TileBufConfigAttr configAttr = getScratchTileConfigAttr(originalScratch, ctx);
  Type elemTy = memTy.getElementType();
  std::string tileTypeStr =
      buildScratchTileTypeString(elemTy, rows, cols, configAttr);

  Value tile = rewriter
                   .create<emitc::VariableOp>(
                       loc, emitc::OpaqueType::get(ctx, tileTypeStr),
                       emitc::OpaqueAttr::get(ctx, ""))
                   .getResult();
  Value scratchAddr = castAddressToU64(rewriter, loc, scratch);
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                       ArrayAttr{}, ArrayAttr{},
                                       ValueRange{tile, scratchAddr});
  return tile;
}

//===----------------------------------------------------------------------===//
// pto.pointer_cast lowering
//===----------------------------------------------------------------------===
static StringRef scatterAtomicTok(pto::ScatterAtomicOp atomic) {
  switch (atomic) {
  case pto::ScatterAtomicOp::None:
    return "pto::ScatterAtomicOp::None";
  case pto::ScatterAtomicOp::Add:
    return "pto::ScatterAtomicOp::Add";
  case pto::ScatterAtomicOp::Max:
    return "pto::ScatterAtomicOp::Max";
  case pto::ScatterAtomicOp::Min:
    return "pto::ScatterAtomicOp::Min";
  }
  llvm_unreachable("unknown ScatterAtomicOp");
}

static StringRef scatterOobTok(pto::ScatterOOB mode) {
  switch (mode) {
  case pto::ScatterOOB::Undefined:
    return "pto::ScatterOOB::Undefined";
  case pto::ScatterOOB::Skip:
    return "pto::ScatterOOB::Skip";
  case pto::ScatterOOB::Clamp:
    return "pto::ScatterOOB::Clamp";
  case pto::ScatterOOB::Wrap:
    return "pto::ScatterOOB::Wrap";
  }
  llvm_unreachable("unknown ScatterOOB");
}

static ArrayAttr buildScatterTemplateArgs(ConversionPatternRewriter &rewriter,
                                          MLIRContext *ctx,
                                          pto::MScatterOp op) {
  SmallVec3<Attribute> templateArgVec;
  const bool rowCoalesce =
      isRowCoalescedMGatherIndexType(op.getSrc().getType(), op.getIdx().getType());
  templateArgVec.push_back(emitc::OpaqueAttr::get(
      ctx, rowCoalesce ? "pto::Coalesce::Row" : "pto::Coalesce::Elem"));
  if (op.getScatterAtomicOp() != pto::ScatterAtomicOp::None ||
      op.getScatterOob() != pto::ScatterOOB::Undefined) {
    templateArgVec.push_back(
        emitc::OpaqueAttr::get(ctx, scatterAtomicTok(op.getScatterAtomicOp())));
    if (op.getScatterOob() != pto::ScatterOOB::Undefined) {
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, scatterOobTok(op.getScatterOob())));
    }
  }
  return rewriter.getArrayAttr(templateArgVec);
}

struct PTOMScatterToMSCATTER : public OpConversionPattern<pto::MScatterOp> {
  using OpConversionPattern<pto::MScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MScatterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Value src = peelUnrealized(adaptor.getSrc());
    Value idx = peelUnrealized(adaptor.getIdx());
    Value mem = peelUnrealized(adaptor.getMem());

    Value memArg = maybeWrapGlobalMemrefAsGlobalTensor(
        rewriter, op.getLoc(), mem, op.getMem().getType(), op.getOperation());
    ArrayAttr templateArgs = buildScatterTemplateArgs(rewriter, ctx, op);

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "MSCATTER",
        ArrayAttr{}, templateArgs,
        ValueRange{memArg, src, idx});

    rewriter.eraseOp(op);
    return success();
  }
};
static void populatePTOToEmitCPatterns(RewritePatternSet &patterns,
                                       TypeConverter &typeConverter,
                                       MLIRContext *ctx,
                                       PTOArch targetArch) {
  populatePTOToEmitCArithPatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCRuntimeOpPatterns(patterns, typeConverter, ctx, targetArch);
  populatePTOToEmitCMemoryOpPatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCTilePatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCSimpleOpPatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCTileMaterializationPatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCSyncPatterns(patterns, typeConverter, ctx, targetArch);
  patterns.add<FuncToEmitC>(typeConverter, ctx);
  populatePTOToEmitCSubviewPatterns(patterns, typeConverter, ctx);
  populatePTOToEmitCKernelOpPatterns(patterns, typeConverter, ctx);
  patterns.add<PTOMScatterToMSCATTER>(typeConverter, ctx);
  patterns.add<PTOMGatherToMGATHER>(typeConverter, ctx);
  populatePTOToEmitCCommPatterns(patterns, typeConverter, ctx, targetArch);
  populatePTOToEmitCControlFlowPatterns(patterns, typeConverter, ctx);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
static constexpr llvm::StringLiteral kCommIncludePreamble = R"cpp(
#ifndef PIPE_FIX
#define PIPE_FIX PIPE_M
#endif
)cpp";

static constexpr llvm::StringLiteral kGlobalTensorDataHelper = R"cpp(
template <typename Tensor>
static AICORE inline auto PTOAS__GLOBAL_TENSOR_DATA(Tensor &tensor)
    -> decltype(tensor.data()) {
  return tensor.data();
}
)cpp";

static constexpr llvm::StringLiteral kEventIdArrayHelper = R"cpp(
template <int N>
struct PTOAS_EventIdArray {
  static_assert(N > 0, "PTOAS_EventIdArray requires a positive static size");
  int32_t data[N] = {};

  AICORE inline int32_t &operator[](int32_t idx) { return data[idx]; }
  AICORE inline const int32_t &operator[](int32_t idx) const { return data[idx]; }
};
)cpp";

static constexpr llvm::StringLiteral kTRandomHelper = R"cpp(
template <uint16_t Rounds, typename DstTile>
static AICORE inline void PTOAS__TRANDOM(
    DstTile &dst, uint32_t key0, uint32_t key1, uint32_t counter0,
    uint32_t counter1, uint32_t counter2, uint32_t counter3) {
  TRandomKey key = {key0, key1};
  TRandomCounter counter = {counter0, counter1, counter2, counter3};
  TRANDOM<Rounds>(dst, key, counter);
}
)cpp";

static constexpr llvm::StringLiteral kAutoSyncTailHelper = R"cpp(
enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}
)cpp";

static constexpr llvm::StringLiteral kBitcastHelper = R"cpp(
template <typename To, typename From>
static inline To ptoas_bitcast(From from) {
  static_assert(sizeof(To) == sizeof(From), "ptoas_bitcast: size mismatch");
  To to;
  __builtin_memcpy(&to, &from, sizeof(To));
  return to;
}
)cpp";

struct EmitPTOManualPass
    : public PassWrapper<EmitPTOManualPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitPTOManualPass)

  struct ModuleHelperNeeds {
    bool needsEventIdArrayHelper = false;
    bool needsTRandomHelper = false;
    bool needsGlobalTensorDataHelper = false;
    bool needsCommInclude = false;
    bool needsBitcastHelper = false;
  };

  PTOArch targetArch;

  EmitPTOManualPass() : targetArch(PTOArch::A3) {}

  explicit EmitPTOManualPass(PTOArch arch) : targetArch(arch) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect, func::FuncDialect, arith::ArithDialect,
                    memref::MemRefDialect, affine::AffineDialect,
                    mlir::cf::ControlFlowDialect, mlir::pto::PTODialect>();
  }

  static void emitCommIncludePreamble(OpBuilder &builder, Location loc) {
    builder.create<emitc::VerbatimOp>(loc, builder.getStringAttr(kCommIncludePreamble));
    builder.create<emitc::IncludeOp>(
        loc, "pto/comm/pto_comm_inst.hpp", /*is_standard_include=*/false);
  }

  static void emitGlobalTensorDataHelper(OpBuilder &builder, Location loc) {
    builder.create<emitc::VerbatimOp>(
        loc, builder.getStringAttr(kGlobalTensorDataHelper));
  }

  static void emitEventIdArrayHelper(OpBuilder &builder, Location loc) {
    builder.create<emitc::VerbatimOp>(loc,
                                      builder.getStringAttr(kEventIdArrayHelper));
  }

  static void emitTRandomHelper(OpBuilder &builder, Location loc) {
    builder.create<emitc::VerbatimOp>(loc, builder.getStringAttr(kTRandomHelper));
  }

  static void emitAutoSyncTailHelper(OpBuilder &builder, Location loc) {
    builder.create<emitc::VerbatimOp>(loc,
                                      builder.getStringAttr(kAutoSyncTailHelper));
  }

  static void emitBitcastHelper(OpBuilder &builder, Location loc) {
    builder.create<emitc::VerbatimOp>(loc, builder.getStringAttr(kBitcastHelper));
  }

  LogicalResult validateA3SyncRequirements(ModuleOp mop) const {
    if (targetArch != PTOArch::A3)
      return success();
    bool hasMissingSetFFTs = false;
    for (auto func : mop.getOps<func::FuncOp>()) {
      if (!hasInterCoreSyncOp(func) || hasSetFFTsOp(func))
        continue;
      hasMissingSetFFTs = true;
      func.emitError()
          << "A3 inter-core sync requires explicit `pto.set_ffts` in the "
             "same function when using `pto.sync.set`/`pto.sync.wait`";
    }
    return success(!hasMissingSetFFTs);
  }

  static ModuleHelperNeeds analyzeModuleHelperNeeds(ModuleOp mop) {
    ModuleHelperNeeds needs;
    mop.walk([&needs](Operation *op) {
      needs.needsEventIdArrayHelper =
          needs.needsEventIdArrayHelper ||
          isa<mlir::pto::DeclareEventIdArrayOp>(op);
      needs.needsTRandomHelper =
          needs.needsTRandomHelper || isa<mlir::pto::TRandomOp>(op);
      needs.needsGlobalTensorDataHelper =
          needs.needsGlobalTensorDataHelper ||
          isa<mlir::pto::PartitionViewOp>(op);
      needs.needsCommInclude =
          needs.needsCommInclude ||
          isa<mlir::pto::BuildAsyncSessionOp, mlir::pto::TPutAsyncOp,
              mlir::pto::TGetAsyncOp, mlir::pto::TPrefetchAsyncOp,
              mlir::pto::WaitAsyncEventOp, mlir::pto::TestAsyncEventOp,
              mlir::pto::TPutOp, mlir::pto::TGetOp, mlir::pto::TNotifyOp,
              mlir::pto::TWaitOp, mlir::pto::TTestOp, mlir::pto::TBroadcastOp,
              mlir::pto::CommTGatherOp, mlir::pto::CommTScatterOp,
              mlir::pto::TReduceOp>(op);
      needs.needsBitcastHelper =
          needs.needsBitcastHelper ||
          isa<arith::BitcastOp, arith::MaximumFOp, arith::MinimumFOp>(op);
    });
    return needs;
  }

  static void insertModulePreamble(ModuleOp mop, MLIRContext *ctx,
                                   const ModuleHelperNeeds &needs) {
    auto loc = mop->getLoc();
    OpBuilder builder(ctx);
    builder.setInsertionPointToStart(mop.getBody());
    builder.create<emitc::IncludeOp>(loc, "pto/pto-inst.hpp",
                                     /*is_standard_include=*/false);
    if (needs.needsCommInclude)
      emitCommIncludePreamble(builder, loc);
    builder.create<emitc::VerbatimOp>(loc, builder.getStringAttr("using namespace pto;"));
    if (needs.needsGlobalTensorDataHelper)
      emitGlobalTensorDataHelper(builder, loc);
    if (needs.needsEventIdArrayHelper)
      emitEventIdArrayHelper(builder, loc);
    if (needs.needsTRandomHelper)
      emitTRandomHelper(builder, loc);
    emitAutoSyncTailHelper(builder, loc);
    if (needs.needsBitcastHelper)
      emitBitcastHelper(builder, loc);
  }

  static LogicalResult runSCFTypePreconversion(
      ModuleOp mop, PTOToEmitCTypeConverter &typeConverter) {
    MLIRContext *ctx = mop.getContext();
    RewritePatternSet scfTypePatterns(ctx);
    ConversionTarget scfTypeTarget(*ctx);
    scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, scfTypePatterns, scfTypeTarget);
    scfTypeTarget.markUnknownOpDynamicallyLegal(
        [](Operation *) { return true; });
    if (failed(applyPartialConversion(mop, scfTypeTarget,
                                      std::move(scfTypePatterns)))) {
      mop.emitError("failed to reconcile SCF structural types");
      return failure();
    }
    return success();
  }

  static ConversionTarget buildMainConversionTarget(
      ModuleOp mop, PTOToEmitCTypeConverter &typeConverter) {
    ConversionTarget target(*mop.getContext());
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<pto::PTODialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<mlir::scf::SCFDialect>();
    target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
        [&typeConverter](Operation *op) {
          return isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                                  typeConverter);
        });
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalOp<func::ReturnOp>();
    target.addIllegalOp<func::FuncOp>();
    target.addIllegalOp<func::CallOp>();
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalOp<ModuleOp>();
    return target;
  }

  static std::unique_ptr<DataFlowSolver> buildAnalysisSolver(ModuleOp mop) {
    auto solver = std::make_unique<DataFlowSolver>();
    solver->load<dataflow::DeadCodeAnalysis>();
    solver->load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver->initializeAndRun(mop)))
      return {};
    return solver;
  }

  LogicalResult runMainConversion(ModuleOp mop, MLIRContext *ctx,
                                  PTOToEmitCTypeConverter &typeConverter) const {
    auto solver = buildAnalysisSolver(mop);
    if (!solver)
      return failure();
    ConversionTarget target = buildMainConversionTarget(mop, typeConverter);
    RewritePatternSet patterns(ctx);
    populatePTOToEmitCPatterns(patterns, typeConverter, ctx, targetArch);
    if (failed(applyPartialConversion(mop, target, std::move(patterns)))) {
      llvm::errs() << "Conversion FAILED! Rolling back executed.\n";
      return failure();
    }
    return success();
  }

  static bool isEmitCTileLikeType(Type ty) {
    auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty);
    if (!opaqueTy)
      return false;
    StringRef value = opaqueTy.getValue();
    return value.contains("Tile<") || value.contains("ConvTile<");
  }

  static LogicalResult lowerUnrealizedCast(
      UnrealizedConversionCastOp cast,
      SmallVectorImpl<UnrealizedConversionCastOp> &castsToErase) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
      cast.emitError() << "unsupported unrealized_conversion_cast shape";
      return failure();
    }

    Value input = cast.getOperand(0);
    Value output = cast.getResult(0);
    Type inTy = input.getType();
    Type outTy = output.getType();
    if (output.use_empty()) {
      castsToErase.push_back(cast);
      return success();
    }
    if (inTy == outTy) {
      output.replaceAllUsesWith(input);
      castsToErase.push_back(cast);
      return success();
    }
    if (isEmitCPointerLikeType(inTy) && isa<BaseMemRefType>(outTy)) {
      output.replaceAllUsesWith(input);
      castsToErase.push_back(cast);
      return success();
    }
    if (isEmitCTileLikeType(inTy) && isa<pto::TileBufType>(outTy)) {
      output.replaceAllUsesWith(input);
      castsToErase.push_back(cast);
      return success();
    }
    if (emitc::isSupportedEmitCType(inTy) && emitc::isSupportedEmitCType(outTy)) {
      OpBuilder builder(cast);
      auto converted = builder.create<emitc::CastOp>(cast.getLoc(), outTy, input);
      output.replaceAllUsesWith(converted.getResult());
      castsToErase.push_back(cast);
      return success();
    }

    cast.emitError() << "cannot lower unrealized_conversion_cast(" << inTy
                     << " -> " << outTy << ") to emitc.cast";
    return failure();
  }

  static LogicalResult cleanupUnrealizedCasts(ModuleOp mop) {
    llvm::SmallVector<UnrealizedConversionCastOp> castsToErase;
    bool castCleanupFailed = false;
    mop.walk([&castCleanupFailed,
              &castsToErase](UnrealizedConversionCastOp cast) {
      if (castCleanupFailed)
        return;
      castCleanupFailed =
          failed(lowerUnrealizedCast(cast, castsToErase));
    });
    for (auto cast : castsToErase)
      cast.erase();
    return success(!castCleanupFailed);
  }

  static void sinkVariableCasts(ModuleOp mop) {
    SmallVector<emitc::CastOp> castOpsToSink;
    mop.walk([&castOpsToSink](emitc::CastOp castOp) {
      if (castOp.getSource().getDefiningOp<emitc::VariableOp>())
        castOpsToSink.push_back(castOp);
    });

    for (emitc::CastOp castOp : castOpsToSink) {
      Value src = castOp.getSource();
      Type dstTy = castOp.getResult().getType();
      Value oldRes = castOp.getResult();
      for (OpOperand &use : llvm::make_early_inc_range(oldRes.getUses())) {
        Operation *user = use.getOwner();
        OpBuilder b(user);
        b.setInsertionPoint(user);
        auto newCast = b.create<emitc::CastOp>(castOp.getLoc(), dstTy, src);
        use.set(newCast.getResult());
      }
      castOp.erase();
    }
  }

  static void fixForInductionVarTypes(ModuleOp mop) {
    mop.walk([](emitc::ForOp forOp) {
      Type boundTy = forOp.getLowerBound().getType();
      BlockArgument iv = forOp.getBody()->getArgument(0);
      if (iv.getType() != boundTy)
        iv.setType(boundTy);
    });
  }

  static void eraseDeadTileVariables(ModuleOp mop) {
    llvm::SmallVector<emitc::VariableOp> deadVars;
    mop.walk([&deadVars](emitc::VariableOp varOp) {
      bool isRead = false;
      for (Operation *user : varOp.getResult().getUsers()) {
        if (auto call = dyn_cast<emitc::CallOpaqueOp>(user)) {
          if (call.getCallee() == "TASSIGN" && call.getOperand(0) == varOp.getResult())
            continue;
        }
        isRead = true;
        break;
      }
      if (!isRead)
        deadVars.push_back(varOp);
    });

    for (auto varOp : deadVars) {
      llvm::SmallVector<Operation *> usersToErase;
      for (Operation *user : varOp.getResult().getUsers())
        usersToErase.push_back(user);
      for (auto user : usersToErase)
        user->erase();
      varOp.erase();
    }
  }

  static void eraseDeadConstants(ModuleOp mop) {
    llvm::SmallVector<emitc::ConstantOp> deadConsts;
    mop.walk([&deadConsts](emitc::ConstantOp constOp) {
      if (constOp.getResult().use_empty())
        deadConsts.push_back(constOp);
    });
    for (auto constOp : deadConsts)
      constOp.erase();
  }

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "DEBUG: Start PTOToEmitC Pass\n");
    MLIRContext *ctx = &getContext();
    ModuleOp mop = getOperation();
    if (failed(pto::validatePTOEntryFunctions(mop)))
      return signalPassFailure();
    pto::annotatePTOEntryFunctions(mop);
    if (failed(validateA3SyncRequirements(mop)))
      return signalPassFailure();

    ModuleHelperNeeds helperNeeds = analyzeModuleHelperNeeds(mop);
    insertModulePreamble(mop, ctx, helperNeeds);
    if (failed(runPTOToEmitCSCFPreLowering(mop, ctx)))
      return signalPassFailure();

    PTOToEmitCTypeConverter typeConverter(ctx, targetArch);
    if (failed(runSCFTypePreconversion(mop, typeConverter)))
      return signalPassFailure();
    if (failed(runMainConversion(mop, ctx, typeConverter)))
      return signalPassFailure();
    if (failed(cleanupUnrealizedCasts(mop)))
      return signalPassFailure();

    sinkVariableCasts(mop);
    fixForInductionVarTypes(mop);
    eraseDeadTileVariables(mop);
    eraseDeadConstants(mop);
  }
  };
} // namespace

std::unique_ptr<Pass> mlir::pto::createEmitPTOManualPass() {
  return std::make_unique<EmitPTOManualPass>();
}

std::unique_ptr<Pass> mlir::pto::createEmitPTOManualPass(PTOArch arch) {
  return std::make_unique<EmitPTOManualPass>(arch);
}

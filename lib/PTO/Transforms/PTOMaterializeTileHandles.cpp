// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOMaterializeTileHandles.cpp -------------------------------------===//
//===----------------------------------------------------------------------===//
//
// Reintroduce explicit tile_buf handles after memory planning/sync have used
// memref IR. EmitC can then lower tile operations from tile-typed operands
// instead of rediscovering tile metadata from every memref use.

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <memory>
#include <optional>
#include <utility>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOMATERIALIZETILEHANDLES
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace mlir {
namespace pto {
namespace {

static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";
constexpr unsigned kTileHandleAttrInlineCapacity = 2;
constexpr unsigned kLoopOpInlineCapacity = 8;
constexpr unsigned kAnchorInlineCapacity = 32;
constexpr unsigned kDeadBindInlineCapacity = 16;
constexpr size_t kTileRank2D = 2;
constexpr unsigned kFirstOperandIndex = 0;
constexpr unsigned kSecondOperandIndex = 1;
constexpr unsigned kThirdOperandIndex = 2;
constexpr unsigned kFourthOperandIndex = 3;

template <typename T>
using SmallVec2 = SmallVector<T, kTileHandleAttrInlineCapacity>;
template <typename T>
using SmallVec8 = SmallVector<T, kLoopOpInlineCapacity>;
template <typename T>
using SmallVec16 = SmallVector<T, kDeadBindInlineCapacity>;
template <typename T>
using SmallVec32 = SmallVector<T, kAnchorInlineCapacity>;

struct TileHandleMetadata {
  Value source;
  Value validRow;
  Value validCol;
  TileBufConfigAttr config;
  bool explicitConfig = false;
  SmallVec2<NamedAttribute> attrs;
};

static bool isLocalTileMemRef(Type type) {
  auto memTy = dyn_cast<MemRefType>(type);
  if (!memTy || memTy.getRank() != static_cast<int64_t>(kTileRank2D))
    return false;

  auto asAttr = dyn_cast_or_null<AddressSpaceAttr>(memTy.getMemorySpace());
  if (!asAttr)
    return false;

  switch (asAttr.getAddressSpace()) {
  case AddressSpace::GM:
  case AddressSpace::Zero:
    return false;
  case AddressSpace::VEC:
  case AddressSpace::MAT:
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
  case AddressSpace::ACC:
  case AddressSpace::BIAS:
  case AddressSpace::SCALING:
    return true;
  }
  return false;
}

static bool shouldMaterializeOperand(Operation *owner) {
  if (isa<AllocTileOp, MaterializeTileOp, BindTileOp, PointerCastOp>(owner))
    return false;

  StringRef name = owner->getName().getStringRef();
  if (name == "pto.set_validshape")
    return true;
  if (name == "pto.get_validshape")
    return true;
  if (name == "pto.build_async_session")
    return true;
  if (!name.consume_front("pto."))
    return false;
  return name.starts_with("t");
}

static bool shouldMaterializeYieldOperand(Operation *owner) {
  return isa<scf::YieldOp>(owner);
}

static bool hasStringAttr(ArrayRef<NamedAttribute> attrs, StringRef name,
                          StringRef value) {
  return llvm::any_of(attrs, [name, value](NamedAttribute attr) {
    if (attr.getName().getValue() != name)
      return false;
    auto strAttr = dyn_cast<StringAttr>(attr.getValue());
    return strAttr && strAttr.getValue() == value;
  });
}

static bool hasAttr(ArrayRef<NamedAttribute> attrs, StringRef name) {
  return llvm::any_of(attrs, [name](NamedAttribute attr) {
    return attr.getName().getValue() == name;
  });
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

static void copyTileHandleAttrs(Operation *from,
                                SmallVectorImpl<NamedAttribute> &attrs) {
  StringRef names[] = {"pto.view_semantics", kForceDynamicValidShapeAttrName};
  for (StringRef name : names) {
    if (Attribute attr = from->getAttr(name))
      attrs.push_back(NamedAttribute(StringAttr::get(from->getContext(), name),
                                     attr));
  }
}

static std::optional<AddressSpace> getAddressSpace(Type type) {
  if (auto memTy = dyn_cast<MemRefType>(type)) {
    if (auto asAttr =
            dyn_cast_or_null<AddressSpaceAttr>(memTy.getMemorySpace()))
      return asAttr.getAddressSpace();
    return std::nullopt;
  }
  if (auto tileTy = dyn_cast<TileBufType>(type)) {
    if (auto asAttr =
            dyn_cast_or_null<AddressSpaceAttr>(tileTy.getMemorySpace()))
      return asAttr.getAddressSpace();
  }
  return std::nullopt;
}

static bool isA5Target(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return false;
  if (auto arch = module->getAttrOfType<StringAttr>("pto.target_arch")) {
    if (arch.getValue().equals_insensitive("a5"))
      return true;
  }
  if (auto spec = module->getAttrOfType<StringAttr>("pto.device-spec")) {
    StringRef value = spec.getValue();
    if (value.starts_with("Ascend950") || value.starts_with("Ascend910_95"))
      return true;
  }
  return false;
}

static TileBufConfigAttr makeTileConfig(MLIRContext *ctx, BLayout bl,
                                        SLayout sl) {
  Builder builder(ctx);
  return TileBufConfigAttr::get(
      ctx, BLayoutAttr::get(ctx, bl), SLayoutAttr::get(ctx, sl),
      builder.getI32IntegerAttr(kFractalSize512),
      PadValueAttr::get(ctx, PadValue::Null),
      CompactModeAttr::get(ctx, CompactMode::Null));
}

static void inferConfigForMaterializedUse(Operation *owner, unsigned operandNo,
                                          Type operandType,
                                          TileHandleMetadata &meta,
                                          MLIRContext *ctx) {
  if (meta.explicitConfig)
    return;

  auto colRow = [ctx]() {
    return makeTileConfig(ctx, BLayout::ColMajor, SLayout::RowMajor);
  };
  auto rowCol = [ctx]() {
    return makeTileConfig(ctx, BLayout::RowMajor, SLayout::ColMajor);
  };
  if (isa<TMatmulOp>(owner)) {
    if (!isA5Target(owner))
      return;
    if (operandNo == kFirstOperandIndex || operandNo == kThirdOperandIndex)
      meta.config = colRow();
    else if (operandNo == kSecondOperandIndex)
      meta.config = rowCol();
    return;
  }

  if (isa<TMatmulAccOp>(owner)) {
    if (!isA5Target(owner))
      return;
    if (operandNo == kFirstOperandIndex || operandNo == kSecondOperandIndex ||
        operandNo == kFourthOperandIndex)
      meta.config = colRow();
    else if (operandNo == kThirdOperandIndex)
      meta.config = rowCol();
    return;
  }

  if (isa<TInsertOp>(owner)) {
    if (operandNo != kFirstOperandIndex && operandNo != kFourthOperandIndex)
      return;
    auto as = getAddressSpace(operandType);
    if (!as)
      return;
    if (*as == AddressSpace::ACC || *as == AddressSpace::MAT)
      meta.config = colRow();
  }
}

static TileHandleMetadata getTileHandleMetadata(Value value,
                                                MLIRContext *ctx) {
  TileHandleMetadata meta;
  meta.source = value;
  meta.config = TileBufConfigAttr::getDefault(ctx);
  if (auto bind = value.getDefiningOp<BindTileOp>()) {
    meta.source = bind.getSource();
    meta.validRow = bind.getValidRow();
    meta.validCol = bind.getValidCol();
    meta.config = bind.getConfig();
    meta.explicitConfig = true;
    copyTileHandleAttrs(bind, meta.attrs);
    return meta;
  }

  if (auto cast = value.getDefiningOp<PointerCastOp>()) {
    meta.validRow = cast.getValidRow();
    meta.validCol = cast.getValidCol();
    if (auto config = cast.getConfig()) {
      meta.config = *config;
      meta.explicitConfig = true;
    }
    copyTileHandleAttrs(cast, meta.attrs);
    return meta;
  }

  return meta;
}

static int32_t getConfigI32Value(Attribute attr) {
  if (auto enumAttr = dyn_cast<IntegerAttr>(attr))
    return static_cast<int32_t>(enumAttr.getInt());
  if (auto blAttr = dyn_cast<BLayoutAttr>(attr))
    return static_cast<int32_t>(blAttr.getValue());
  if (auto slAttr = dyn_cast<SLayoutAttr>(attr))
    return static_cast<int32_t>(slAttr.getValue());
  return 0;
}

static FailureOr<std::pair<int64_t, int64_t>>
getBoxedTileInnerShape(TileBufConfigAttr configAttr, Type elemTy, int32_t slVal) {
  int32_t fractal = kFractalSize512;
  if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
    fractal = static_cast<int32_t>(frAttr.getInt());

  unsigned elemBytes = pto::getPTOStorageElemByteSize(elemTy);
  if (elemBytes == 0)
    return failure();

  switch (fractal) {
  case kFractalSize1024:
    return std::make_pair<int64_t, int64_t>(kFractalSize16, kFractalSize16);
  case kFractalSize32:
    return std::make_pair<int64_t, int64_t>(kFractalSize16,
                                            kFractalSize32 / kFractalSize16);
  case kFractalSize512:
    if (slVal == static_cast<int32_t>(SLayout::RowMajor))
      return std::make_pair<int64_t, int64_t>(kFractalSize16,
                                              kFractalSize32 / elemBytes);
    if (slVal == static_cast<int32_t>(SLayout::ColMajor))
      return std::make_pair<int64_t, int64_t>(kFractalSize32 / elemBytes,
                                              kFractalSize16);
    return failure();
  default:
    return failure();
  }
}

static bool setDenseTilePointerStrides(int32_t blVal, int64_t rows,
                                       int64_t cols, int64_t &rowStride,
                                       int64_t &colStride) {
  if (blVal == 1) {
    rowStride = 1;
    colStride = rows;
  } else {
    rowStride = cols;
    colStride = 1;
  }
  return true;
}

static bool setBoxedTilePointerStrides(int32_t blVal, int32_t slVal,
                                       int64_t rows, int64_t cols,
                                       int64_t innerRows, int64_t innerCols,
                                       int64_t &rowStride,
                                       int64_t &colStride) {
  if (blVal == 1) {
    if (slVal != 1)
      return false;
    rowStride = innerCols;
    colStride = rows;
    return true;
  }
  rowStride = cols;
  colStride = innerRows;
  return true;
}

static bool getTilePointerStrides(TileBufConfigAttr configAttr, Type elemTy,
                                  int64_t rows, int64_t cols,
                                  int64_t &rowStride, int64_t &colStride) {
  if (rows == ShapedType::kDynamic || cols == ShapedType::kDynamic)
    return false;

  int32_t blVal = getConfigI32Value(configAttr.getBLayout());
  int32_t slVal = getConfigI32Value(configAttr.getSLayout());
  bool boxed = slVal != 0;
  if (!boxed)
    return setDenseTilePointerStrides(blVal, rows, cols, rowStride,
                                      colStride);
  auto innerShape = getBoxedTileInnerShape(configAttr, elemTy, slVal);
  if (failed(innerShape) || innerShape->first <= 0 || innerShape->second <= 0)
    return false;
  return setBoxedTilePointerStrides(blVal, slVal, rows, cols,
                                    innerShape->first, innerShape->second,
                                    rowStride, colStride);
}

static SmallVec2<int64_t>
getMaterializedTileShape(MemRefType memTy, const TileHandleMetadata &meta) {
  SmallVec2<int64_t> shape(memTy.getShape().begin(), memTy.getShape().end());
  if (!hasStringAttr(meta.attrs, "pto.view_semantics", "subview"))
    return shape;

  auto sourceMrTy = dyn_cast_or_null<MemRefType>(meta.source.getType());
  if (!sourceMrTy || sourceMrTy.getRank() < static_cast<int64_t>(kTileRank2D) ||
      !meta.source.getDefiningOp<memref::SubViewOp>())
    return shape;

  int64_t subRows = sourceMrTy.getDimSize(0);
  int64_t subCols = sourceMrTy.getDimSize(1);
  if (pto::isPTOFloat4PackedType(sourceMrTy.getElementType()) ||
      subRows == ShapedType::kDynamic || subCols == ShapedType::kDynamic)
    return shape;

  SmallVector<int64_t> inheritedStrides;
  int64_t inheritedOffset = ShapedType::kDynamic;
  if (failed(getStridesAndOffset(sourceMrTy, inheritedStrides,
                                 inheritedOffset)) ||
      inheritedStrides.size() < kTileRank2D)
    return shape;

  int64_t childRowStride = 0;
  int64_t childColStride = 0;
  if (!getTilePointerStrides(meta.config, sourceMrTy.getElementType(), subRows,
                             subCols, childRowStride, childColStride))
    return shape;
  if (inheritedStrides[0] == childRowStride &&
      inheritedStrides[1] == childColStride) {
    shape[0] = subRows;
    shape[1] = subCols;
  }

  return shape;
}

static TileBufType buildTileTypeFromMemRef(MemRefType memTy,
                                           const TileHandleMetadata &meta,
                                           MLIRContext *ctx) {
  SmallVec2<int64_t> shape = getMaterializedTileShape(memTy, meta);
  SmallVec2<int64_t> validShape(shape.begin(), shape.end());
  bool forceDynamic = hasAttr(meta.attrs, kForceDynamicValidShapeAttrName);
  if (forceDynamic) {
    validShape[0] = ShapedType::kDynamic;
    validShape[1] = ShapedType::kDynamic;
  } else {
    if (meta.validRow)
      validShape[0] = getConstantIndexOrDynamic(meta.validRow);
    if (meta.validCol)
      validShape[1] = getConstantIndexOrDynamic(meta.validCol);
  }

  return TileBufType::get(ctx, shape, memTy.getElementType(),
                          memTy.getMemorySpace(), validShape, meta.config);
}

static bool isMaterializedTileAnchor(Operation *op) {
  return isa<BindTileOp, PointerCastOp>(op);
}

static Value makeI64Constant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIntOp>(loc, value, kPTOI64BitWidth);
}

static Value ensureI64(Value value, OpBuilder &builder, Location loc) {
  if (!value)
    return Value();

  auto i64Ty = builder.getI64Type();
  if (value.getType() == i64Ty)
    return value;
  if (isa<IndexType>(value.getType()))
    return builder.create<arith::IndexCastOp>(loc, i64Ty, value);
  if (auto intTy = dyn_cast<IntegerType>(value.getType())) {
    if (intTy.getWidth() == kPTOI64BitWidth)
      return value;
    if (intTy.getWidth() < kPTOI64BitWidth)
      return builder.create<arith::ExtSIOp>(loc, i64Ty, value);
    return builder.create<arith::TruncIOp>(loc, i64Ty, value);
  }
  return Value();
}

static Value materializeOffset(OpFoldResult ofr, OpBuilder &builder,
                               Location loc) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return makeI64Constant(builder, loc, intAttr.getInt());
    return Value();
  }
  return ensureI64(ofr.get<Value>(), builder, loc);
}

static Value addI64(Value lhs, Value rhs, OpBuilder &builder, Location loc) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return builder.create<arith::AddIOp>(loc, lhs, rhs);
}

static Value mulI64(Value lhs, int64_t rhs, OpBuilder &builder, Location loc) {
  if (!lhs)
    return Value();
  if (rhs == 0)
    return makeI64Constant(builder, loc, 0);
  if (rhs == 1)
    return lhs;
  return builder.create<arith::MulIOp>(loc, lhs,
                                       makeI64Constant(builder, loc, rhs));
}

static Value computeExplicitAddress(Value value, OpBuilder &builder,
                                    Location loc);

static Value computeSubviewAddress(memref::SubViewOp subview,
                                   OpBuilder &builder, Location loc) {
  Value base = computeExplicitAddress(subview.getSource(), builder, loc);
  if (!base)
    return Value();

  auto sourceTy = dyn_cast<MemRefType>(subview.getSource().getType());
  if (!sourceTy)
    return Value();
  unsigned elemBytes = getPTOStorageElemByteSize(sourceTy.getElementType());
  if (elemBytes == 0)
    return Value();

  SmallVector<int64_t> sourceStrides;
  int64_t sourceOffset = ShapedType::kDynamic;
  if (failed(getStridesAndOffset(sourceTy, sourceStrides, sourceOffset)))
    return Value();

  auto mixedOffsets = subview.getMixedOffsets();
  if (sourceStrides.size() < mixedOffsets.size())
    return Value();

  Value linearOffset;
  for (auto [offsetOfr, stride] :
       llvm::zip_equal(mixedOffsets, ArrayRef<int64_t>(sourceStrides).take_front(
                                         mixedOffsets.size()))) {
    if (stride == ShapedType::kDynamic)
      return Value();
    Value offset = materializeOffset(offsetOfr, builder, loc);
    if (!offset)
      return Value();
    linearOffset = addI64(linearOffset, mulI64(offset, stride, builder, loc),
                          builder, loc);
  }

  if (!linearOffset)
    return base;
  linearOffset = mulI64(linearOffset, elemBytes, builder, loc);
  return builder.create<arith::AddIOp>(loc, base, linearOffset);
}

static Value computeExplicitAddress(Value value, OpBuilder &builder,
                                    Location loc) {
  if (auto bind = value.getDefiningOp<BindTileOp>())
    return computeExplicitAddress(bind.getSource(), builder, loc);
  if (auto cast = value.getDefiningOp<PointerCastOp>()) {
    if (cast.getAddrs().empty())
      return Value();
    return ensureI64(cast.getAddrs().front(), builder, loc);
  }

  if (auto subview = value.getDefiningOp<memref::SubViewOp>())
    return computeSubviewAddress(subview, builder, loc);
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return computeExplicitAddress(cast.getSource(), builder, loc);
  return Value();
}

static bool isControlFlowAddressProducer(Operation *op) {
  if (!op)
    return false;

  StringRef name = op->getName().getStringRef();
  return name == "scf.if" || name == "scf.for" || name == "scf.while" ||
         name == "scf.execute_region" || name == "scf.index_switch";
}

static Value peelAddressSource(Value value) {
  while (true) {
    if (auto bind = value.getDefiningOp<BindTileOp>()) {
      value = bind.getSource();
      continue;
    }

    if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
      value = subview.getSource();
      continue;
    }

    if (auto cast = value.getDefiningOp<memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }

    return value;
  }
}

static bool isFunctionEntryBlockArgument(BlockArgument arg) {
  Operation *parent = arg.getOwner()->getParentOp();
  auto func = dyn_cast_or_null<func::FuncOp>(parent);
  return func && arg.getOwner() == &func.getBody().front();
}

static bool isUnsupportedControlFlowAddress(Value value) {
  value = peelAddressSource(value);
  if (auto arg = dyn_cast<BlockArgument>(value))
    return !isFunctionEntryBlockArgument(arg);
  return isControlFlowAddressProducer(value.getDefiningOp());
}

static void emitMissingExplicitAddressError(Operation *owner, Value value) {
  value = peelAddressSource(value);
  auto diag = owner->emitOpError()
              << "cannot materialize tile handle for local memref because its "
                 "explicit byte address cannot be recovered";
  if (isa<BlockArgument>(value)) {
    diag << "; region block arguments and loop-carried memref values are "
            "unsupported here";
    return;
  }

  Operation *def = value.getDefiningOp();
  if (!def) {
    diag << "; value has no defining op";
    return;
  }

  if (isControlFlowAddressProducer(def)) {
    diag << "; control-flow result '" << def->getName()
         << "' cannot carry a local memref into tile materialization";
    return;
  }

  diag << "; unsupported defining op is '" << def->getName() << "'";
}

static Value lookupMaterializedTileHandle(
    Value value, DenseMap<Value, Value> &tileHandles) {
  if (isa<TileBufType>(value.getType()))
    return value;

  auto it = tileHandles.find(value);
  if (it == tileHandles.end())
    return Value();
  return it->second;
}

static FailureOr<bool>
materializeSCFIfResults(ModuleOp module, DenseMap<Value, Value> &tileHandles) {
  bool changed = false;

  SmallVec8<scf::IfOp> ifOps;
  module.walk([&ifOps](scf::IfOp ifOp) { ifOps.push_back(ifOp); });

  for (scf::IfOp ifOp : llvm::reverse(ifOps)) {
    if (ifOp.getNumResults() == 0)
      continue;

    auto thenYield = dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYield = dyn_cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    if (!thenYield || !elseYield)
      continue;

    for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
      if (!isLocalTileMemRef(result.getType()))
        continue;

      Value thenTile =
          lookupMaterializedTileHandle(thenYield.getOperand(idx), tileHandles);
      Value elseTile =
          lookupMaterializedTileHandle(elseYield.getOperand(idx), tileHandles);
      if (!thenTile || !elseTile)
        continue;
      if (thenTile.getType() != elseTile.getType()) {
        ifOp.emitOpError()
            << "cannot materialize tile result #" << idx
            << " because branch tile types differ: " << thenTile.getType()
            << " vs " << elseTile.getType();
        return failure();
      }

      Type tileTy = thenTile.getType();
      thenYield->setOperand(idx, thenTile);
      elseYield->setOperand(idx, elseTile);
      result.setType(tileTy);
      tileHandles[result] = result;
      changed = true;
    }
  }

  return changed;
}

static FailureOr<bool>
materializeSCFForResults(ModuleOp module, DenseMap<Value, Value> &tileHandles) {
  bool changed = false;

  SmallVec8<scf::ForOp> forOps;
  module.walk([&forOps](scf::ForOp forOp) { forOps.push_back(forOp); });

  for (scf::ForOp forOp : llvm::reverse(forOps)) {
    if (forOp.getNumResults() == 0)
      continue;

    auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield)
      continue;

    for (auto [idx, result] : llvm::enumerate(forOp.getResults())) {
      if (!isLocalTileMemRef(result.getType()))
        continue;

      Value initTile =
          lookupMaterializedTileHandle(forOp.getInitArgs()[idx], tileHandles);
      if (!initTile)
        continue;

      BlockArgument iterArg = forOp.getRegionIterArg(idx);
      Value yieldValue = yield.getOperand(idx);
      Value yieldTile = lookupMaterializedTileHandle(yieldValue, tileHandles);
      bool yieldIsIterArg = !yieldTile && yieldValue == iterArg;
      if (yieldIsIterArg)
        yieldTile = iterArg;
      if (!yieldTile)
        continue;

      Type yieldTy = yieldIsIterArg ? initTile.getType() : yieldTile.getType();
      if (initTile.getType() != yieldTy) {
        forOp.emitOpError()
            << "cannot materialize tile result #" << idx
            << " because init/yield tile types differ: " << initTile.getType()
            << " vs " << yieldTy;
        return failure();
      }

      Type tileTy = initTile.getType();
      forOp->setOperand(forOp.getNumControlOperands() + idx, initTile);
      iterArg.setType(tileTy);
      yield->setOperand(idx, yieldTile);
      result.setType(tileTy);
      tileHandles[iterArg] = iterArg;
      tileHandles[result] = result;
      changed = true;
    }
  }

  return changed;
}

static LogicalResult
materializeControlFlowTileResults(ModuleOp module,
                                  DenseMap<Value, Value> &tileHandles) {
  bool changed = false;
  do {
    changed = false;

    FailureOr<bool> ifChanged =
        materializeSCFIfResults(module, tileHandles);
    if (failed(ifChanged))
      return failure();
    changed = changed || *ifChanged;

    FailureOr<bool> forChanged =
        materializeSCFForResults(module, tileHandles);
    if (failed(forChanged))
      return failure();
    changed = changed || *forChanged;
  } while (changed);
  return success();
}

static Value getAllocValidOperand(TileBufType tileTy, Value operand,
                                  unsigned dim, OpBuilder &builder,
                                  Location loc) {
  auto validShape = tileTy.getValidShape();
  if (validShape.size() <= dim || validShape[dim] >= 0)
    return Value();
  if (operand)
    return operand;

  auto shape = tileTy.getShape();
  if (shape.size() > dim && shape[dim] != ShapedType::kDynamic)
    return builder.create<arith::ConstantIndexOp>(loc, shape[dim]);
  return Value();
}

static Attribute getAttr(ArrayRef<NamedAttribute> attrs, StringRef name) {
  for (NamedAttribute attr : attrs) {
    if (attr.getName().getValue() == name)
      return attr.getValue();
  }
  return {};
}

static void copyMaterializedTileAttrs(ArrayRef<NamedAttribute> attrs,
                                      Operation *to) {
  if (Attribute attr = getAttr(attrs, kForceDynamicValidShapeAttrName))
    to->setAttr(kForceDynamicValidShapeAttrName, attr);
}

static void updateResultTypesAfterMaterializingOperand(Operation *op,
                                                       unsigned operandNo,
                                                       Type tileTy) {
  if (auto tassign = dyn_cast<TAssignOp>(op)) {
    if (operandNo == 0)
      tassign.getResult().setType(tileTy);
  }
}

static bool isTileViewSemantics(StringAttr viewSemantics) {
  return viewSemantics && (viewSemantics.getValue() == "treshape" ||
                           viewSemantics.getValue() == "bitcast");
}

static SmallVector<OpOperand *> collectMaterializedTileUses(Value anchoredValue) {
  SmallVector<OpOperand *> usesToRewrite;
  for (OpOperand &use : anchoredValue.getUses()) {
    if (shouldMaterializeOperand(use.getOwner()) ||
        shouldMaterializeYieldOperand(use.getOwner())) {
      usesToRewrite.push_back(&use);
    }
  }
  return usesToRewrite;
}

static Value createMaterializedTileFromSource(
    Operation *anchor, Type tileTy, StringAttr viewSemantics, Value sourceTile,
    OpBuilder &builder) {
  if (!(sourceTile && viewSemantics))
    return Value();
  if (viewSemantics.getValue() == "treshape") {
    return builder.create<TReshapeOp>(anchor->getLoc(), tileTy, sourceTile)
        .getResult();
  }
  if (viewSemantics.getValue() == "bitcast") {
    return builder.create<BitcastOp>(anchor->getLoc(), tileTy, sourceTile)
        .getResult();
  }
  return Value();
}

static Value createMaterializedAllocTile(Operation *anchor, Value anchoredValue,
                                         Type tileTy,
                                         const TileHandleMetadata &meta,
                                         OpBuilder &builder,
                                         bool &failedMaterialization) {
  Value addr = computeExplicitAddress(anchoredValue, builder, anchor->getLoc());
  if (!addr && isUnsupportedControlFlowAddress(anchoredValue)) {
    emitMissingExplicitAddressError(anchor, anchoredValue);
    failedMaterialization = true;
    return Value();
  }
  auto alloc = builder.create<AllocTileOp>(
      anchor->getLoc(), tileTy, addr ? addr : Value(),
      getAllocValidOperand(cast<TileBufType>(tileTy), meta.validRow, 0, builder,
                           anchor->getLoc()),
      getAllocValidOperand(cast<TileBufType>(tileTy), meta.validCol, 1, builder,
                           anchor->getLoc()));
  copyMaterializedTileAttrs(meta.attrs, alloc);
  return alloc.getResult();
}

static void rewriteMaterializedTileUses(ArrayRef<OpOperand *> usesToRewrite,
                                        Value materialized, Type tileTy) {
  for (OpOperand *use : usesToRewrite) {
    Operation *owner = use->getOwner();
    unsigned operandNo = use->getOperandNumber();
    use->set(materialized);
    updateResultTypesAfterMaterializingOperand(owner, operandNo, tileTy);
  }
}

static Value materializeAnchorResult(Operation *anchor, Value anchoredValue,
                                     OpBuilder &builder, MLIRContext *ctx,
                                     DenseMap<Value, Value> &tileHandles,
                                     const DenseSet<Value> &mustMaterialize,
                                     bool &failedMaterialization) {
  auto memTy = dyn_cast<MemRefType>(anchoredValue.getType());
  if (!memTy || !isLocalTileMemRef(memTy))
    return Value();

  SmallVector<OpOperand *> usesToRewrite =
      collectMaterializedTileUses(anchoredValue);
  TileHandleMetadata meta = getTileHandleMetadata(anchoredValue, ctx);
  auto viewSemantics = dyn_cast_or_null<StringAttr>(
      getAttr(meta.attrs, "pto.view_semantics"));
  bool isTileView = isTileViewSemantics(viewSemantics);
  if (usesToRewrite.empty() && !isTileView &&
      !mustMaterialize.contains(anchoredValue))
    return Value();

  for (OpOperand *use : usesToRewrite)
    inferConfigForMaterializedUse(use->getOwner(), use->getOperandNumber(),
                                  anchoredValue.getType(), meta, ctx);
  auto tileTy = buildTileTypeFromMemRef(memTy, meta, ctx);

  builder.setInsertionPointAfter(anchor);
  Value sourceTile = meta.source ? tileHandles.lookup(meta.source) : Value();
  Value materialized = createMaterializedTileFromSource(
      anchor, tileTy, viewSemantics, sourceTile, builder);
  if (!materialized) {
    materialized = createMaterializedAllocTile(anchor, anchoredValue, tileTy,
                                               meta, builder,
                                               failedMaterialization);
    if (!materialized)
      return Value();
  }
  rewriteMaterializedTileUses(usesToRewrite, materialized, tileTy);

  tileHandles[anchoredValue] = materialized;
  return materialized;
}

static SmallVec32<Operation *> collectMaterializedTileAnchors(ModuleOp module) {
  SmallVec32<Operation *> anchors;
  module.walk([&anchors](Operation *op) {
    if (isMaterializedTileAnchor(op))
      anchors.push_back(op);
  });
  return anchors;
}

static DenseSet<Value> collectMustMaterializeSources(ArrayRef<Operation *> anchors,
                                                     MLIRContext *ctx) {
  DenseSet<Value> mustMaterialize;
  for (Operation *anchor : anchors) {
    if (anchor->getNumResults() != 1)
      continue;
    Value anchoredValue = anchor->getResult(0);
    if (!isLocalTileMemRef(anchoredValue.getType()))
      continue;
    TileHandleMetadata meta = getTileHandleMetadata(anchoredValue, ctx);
    auto viewSemantics = dyn_cast_or_null<StringAttr>(
        getAttr(meta.attrs, "pto.view_semantics"));
    if (isTileViewSemantics(viewSemantics) && meta.source)
      mustMaterialize.insert(meta.source);
  }
  return mustMaterialize;
}

static SmallVec32<std::pair<Operation *, unsigned>>
collectTileOperandsToRewrite(ModuleOp module) {
  SmallVec32<std::pair<Operation *, unsigned>> operandsToRewrite;
  module.walk([&operandsToRewrite](Operation *op) {
    if (!shouldMaterializeOperand(op))
      return;
    for (OpOperand &operand : op->getOpOperands()) {
      if (isLocalTileMemRef(operand.get().getType()))
        operandsToRewrite.push_back({op, operand.getOperandNumber()});
    }
  });
  return operandsToRewrite;
}

static void eraseDeadBindTiles(ModuleOp module) {
  bool erasedBind = true;
  while (erasedBind) {
    erasedBind = false;
    SmallVec16<Operation *> deadBinds;
    module.walk([&deadBinds](BindTileOp op) {
      if (op.getResult().use_empty())
        deadBinds.push_back(op);
    });
    for (Operation *op : deadBinds) {
      op->erase();
      erasedBind = true;
    }
  }
}

static bool rewriteMaterializedTileOperand(
    Operation *op, unsigned operandNo, OpBuilder &builder, MLIRContext *ctx,
    DenseMap<Value, Value> &tileHandles, bool &failedMaterialization) {
  Value oldValue = op->getOperand(operandNo);
  if (!isa<MemRefType>(oldValue.getType()))
    return true;
  if (op->getName().getStringRef() == "pto.tassign" && operandNo == 0)
    return true;

  auto memTy = cast<MemRefType>(oldValue.getType());
  TileHandleMetadata meta = getTileHandleMetadata(oldValue, ctx);
  inferConfigForMaterializedUse(op, operandNo, oldValue.getType(), meta, ctx);
  auto tileTy = buildTileTypeFromMemRef(memTy, meta, ctx);

  builder.setInsertionPoint(op);
  Value addr = computeExplicitAddress(oldValue, builder, op->getLoc());
  if (!addr && isUnsupportedControlFlowAddress(oldValue)) {
    emitMissingExplicitAddressError(op, oldValue);
    failedMaterialization = true;
    return false;
  }
  auto alloc = builder.create<AllocTileOp>(
      op->getLoc(), tileTy, addr ? addr : Value(),
      getAllocValidOperand(tileTy, meta.validRow, 0, builder, op->getLoc()),
      getAllocValidOperand(tileTy, meta.validCol, 1, builder, op->getLoc()));
  copyMaterializedTileAttrs(meta.attrs, alloc);
  tileHandles[oldValue] = alloc.getResult();
  op->setOperand(operandNo, alloc.getResult());
  updateResultTypesAfterMaterializingOperand(op, operandNo, tileTy);
  return true;
}

static void materializeTileOperands(
    ModuleOp module, OpBuilder &builder, MLIRContext *ctx,
    DenseMap<Value, Value> &tileHandles, bool &failedMaterialization) {
  for (auto [op, operandNo] : collectTileOperandsToRewrite(module)) {
    if (!rewriteMaterializedTileOperand(op, operandNo, builder, ctx,
                                        tileHandles, failedMaterialization)) {
      continue;
    }
  }
}

struct PTOMaterializeTileHandlesPass
    : public impl::PTOMaterializeTileHandlesBase<
          PTOMaterializeTileHandlesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    OpBuilder builder(ctx);
    DenseMap<Value, Value> tileHandles;
    bool failedMaterialization = false;
    SmallVec32<Operation *> anchors = collectMaterializedTileAnchors(module);
    DenseSet<Value> mustMaterialize = collectMustMaterializeSources(anchors, ctx);

    for (Operation *anchor : anchors) {
      if (anchor->getNumResults() != 1)
        continue;
      materializeAnchorResult(anchor, anchor->getResult(0), builder, ctx,
                              tileHandles, mustMaterialize,
                              failedMaterialization);
    }

    if (failedMaterialization) {
      signalPassFailure();
      return;
    }

    if (failed(materializeControlFlowTileResults(module, tileHandles))) {
      signalPassFailure();
      return;
    }
    materializeTileOperands(module, builder, ctx, tileHandles,
                            failedMaterialization);
    if (failedMaterialization) {
      signalPassFailure();
      return;
    }
    eraseDeadBindTiles(module);
  }
};

} // namespace

std::unique_ptr<Pass> createPTOMaterializeTileHandlesPass() {
  return std::make_unique<PTOMaterializeTileHandlesPass>();
}

} // namespace pto
} // namespace mlir

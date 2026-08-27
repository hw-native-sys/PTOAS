// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- FoldTileBufIntrinsics.cpp ------------------------------------------===//
//
// After TileLang DSL template functions are inlined, the IR contains
// structured-view intrinsics that reference template parameters:
//
// tile_buf family:
//   - pto.tile_buf_addr   → extract pointer address from tile_buf
//   - pto.tile_valid_rows → extract valid row count
//   - pto.tile_valid_cols → extract valid column count
//
// tensor_view family:
//   - pto.tensor_view_addr       → extract pointer from tensor_view
//   - pto.get_tensor_view_dim    → extract dimension size
//   - pto.get_tensor_view_stride → extract dimension stride
//
// This pass resolves them against the concrete values at the call site.
// For tile_buf intrinsics, the active VPTO path folds against addressed
// `pto.alloc_tile` handles produced by the shared tile-handle bridge.
// For tensor_view intrinsics, the pass traces through the full
// unrealized_conversion_cast → memref.subview → memref.reinterpret_cast
// chain to fold directly to constants or SSA operands from the
// reinterpret_cast, without generating intermediate memref.dim /
// memref.extract_strided_metadata ops.
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace pto {
  #define GEN_PASS_DEF_FOLDTILEBUFINTRINSICS
  #include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

enum class FoldIntrinsicMode {
  All,
  ShapeOnly,
  AddrOnly,
};

constexpr llvm::StringLiteral kTileOpValidShapeReadAttr =
    "__pto.tileop_valid_shape_abi";

static FailureOr<FoldIntrinsicMode> parseFoldIntrinsicMode(StringRef mode) {
  if (mode.empty() || mode == "all") {
    return FoldIntrinsicMode::All;
  }
  if (mode == "shape-only") {
    return FoldIntrinsicMode::ShapeOnly;
  }
  if (mode == "addr-only") {
    return FoldIntrinsicMode::AddrOnly;
  }
  return failure();
}

static bool shouldFoldShapeFamily(FoldIntrinsicMode mode) {
  return mode == FoldIntrinsicMode::All || mode == FoldIntrinsicMode::ShapeOnly;
}

static bool shouldFoldAddrFamily(FoldIntrinsicMode mode) {
  return mode == FoldIntrinsicMode::All || mode == FoldIntrinsicMode::AddrOnly;
}

static bool eraseDeadAllocTileOps(func::FuncOp func) {
  SmallVector<pto::AllocTileOp> deadAllocs;
  func.walk([&](pto::AllocTileOp alloc) {
    if (alloc.getResult().use_empty()) {
      deadAllocs.push_back(alloc);
    }
  });

  for (pto::AllocTileOp alloc : llvm::reverse(deadAllocs)) {
    alloc.erase();
  }
  return !deadAllocs.empty();
}

static bool isPTOViewBridgeType(Type type) {
  return isa<pto::PartitionTensorViewType, pto::TensorViewType,
             pto::TileBufType>(type);
}

static bool eraseDeadViewBridgeCasts(func::FuncOp func) {
  SmallVector<UnrealizedConversionCastOp, mlir::pto::kValue8> deadCasts;
  func.walk([&](UnrealizedConversionCastOp castOp) {
    if (!castOp.use_empty() || castOp.getNumOperands() != 1 ||
        castOp.getNumResults() != 1) {
      return;
    }

    Type srcTy = castOp.getOperand(0).getType();
    Type dstTy = castOp.getResult(0).getType();
    if ((isa<MemRefType>(srcTy) && isPTOViewBridgeType(dstTy)) ||
        (isPTOViewBridgeType(srcTy) && isa<MemRefType>(dstTy))) {
      deadCasts.push_back(castOp);
    }
  });

  for (auto castOp : llvm::reverse(deadCasts)) {
    castOp.erase();
  }
  return !deadCasts.empty();
}

static bool eraseDeadMemrefViewOps(func::FuncOp func) {
  SmallVector<Operation *, mlir::pto::kValue8> deadMemrefOps;
  func.walk([&](Operation *op) {
    if ((isa<memref::SubViewOp>(op) ||
         isa<memref::ReinterpretCastOp>(op)) &&
        op->use_empty()) {
      deadMemrefOps.push_back(op);
    }
  });

  for (Operation *op : llvm::reverse(deadMemrefOps)) {
    op->erase();
  }
  return !deadMemrefOps.empty();
}

static bool eraseDeadTensorViewOps(func::FuncOp func) {
  SmallVector<Operation *, mlir::pto::kValue8> deadViewOps;
  func.walk([&](Operation *op) {
    if ((isa<pto::PartitionViewOp>(op) ||
         isa<pto::MakeTensorViewOp>(op)) &&
        op->use_empty()) {
      deadViewOps.push_back(op);
    }
  });

  for (Operation *op : llvm::reverse(deadViewOps)) {
    op->erase();
  }
  return !deadViewOps.empty();
}

static void eraseDeadViewChains(func::FuncOp func) {
  while (eraseDeadViewBridgeCasts(func) || eraseDeadMemrefViewOps(func) ||
         eraseDeadTensorViewOps(func) || eraseDeadAllocTileOps(func)) {
  }
}

struct TileHandleInfo {
  Value addr;
  Value validRow;
  Value validCol;
  pto::TileBufConfigAttr config;
};

static std::pair<Value, Value> findSetValidShapeOverride(Value tileBuf) {
  for (Operation *user : tileBuf.getUsers()) {
    auto setValid = dyn_cast<pto::SetValidShapeOp>(user);
    if (!setValid || setValid.getSource() != tileBuf) {
      continue;
    }
    return {setValid.getValidRow(), setValid.getValidCol()};
  }
  return {Value(), Value()};
}

// Walk through value-conversion casts that preserve the underlying producer.
// ExpandTileOp / PTOInlineLibCall and other passes may wrap tile_buf values in
// UnrealizedConversionCastOp / arith.index_cast / memref.cast when bridging
// tile_buf parameter types. These wrappers must not defeat anchor recovery.
static Value unwrapBridgingCasts(Value v) {
  while (v) {
    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
      break;
    }
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
      if (cast.getNumOperands() == 1 && cast.getNumResults() == 1) {
        v = cast.getOperand(0);
        continue;
      }
      break;
    }
    if (auto cast = dyn_cast<arith::IndexCastOp>(defOp)) {
      v = cast.getIn();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(defOp)) {
      v = cast.getSource();
      continue;
    }
    break;
  }
  return v;
}

static bool isSCFTileCarrier(Value value) {
  if (auto result = dyn_cast<OpResult>(value)) {
    return isa<scf::IfOp, scf::ForOp, arith::SelectOp>(result.getOwner());
  }

  auto blockArg = dyn_cast<BlockArgument>(value);
  return blockArg &&
         isa_and_nonnull<scf::ForOp>(blockArg.getOwner()->getParentOp());
}

/// Returns whether the declared tile is rebound by a TPOP. The VPTO pipe
/// bridge resolves such tiles at runtime from the FIFO slot address returned
/// by the pop, so folding their address here would break address propagation.
static bool isDeclareTileReboundByTPop(pto::DeclareTileOp decl) {
  return llvm::any_of(decl.getResult().getUsers(),
                      [](Operation *user) { return isa<pto::TPopOp>(user); });
}

static std::optional<TileHandleInfo> resolveTileHandle(Value tileBuf,
                                                       Operation *user) {
  if (auto regionResult = dyn_cast<OpResult>(tileBuf)) {
    if (auto fusionRegion =
            dyn_cast<pto::FusionRegionOp>(regionResult.getOwner())) {
      auto yieldOp = dyn_cast<pto::YieldOp>(
          fusionRegion.getBody().front().getTerminator());
      if (!yieldOp) {
        user->emitError("FoldTileBufIntrinsics: pto.fusion_region must "
                        "terminate with pto.yield");
        return std::nullopt;
      }
      unsigned resultIndex = regionResult.getResultNumber();
      if (resultIndex >= yieldOp.getNumOperands()) {
        user->emitError("FoldTileBufIntrinsics: pto.fusion_region result/yield "
                        "sizes are inconsistent");
        return std::nullopt;
      }
      return resolveTileHandle(yieldOp.getOperand(resultIndex), user);
    }
  }

  tileBuf = unwrapBridgingCasts(tileBuf);
  if (auto alloc = tileBuf.getDefiningOp<pto::AllocTileOp>()) {
    auto tileTy = dyn_cast<pto::TileBufType>(alloc.getResult().getType());
    if (!tileTy) {
      user->emitError(
          "FoldTileBufIntrinsics: pto.alloc_tile must produce !pto.tile_buf");
      return std::nullopt;
    }
    return TileHandleInfo{alloc.getAddr(), alloc.getValidRow(),
                          alloc.getValidCol(), tileTy.getConfigAttr()};
  }

  if (auto decl = tileBuf.getDefiningOp<pto::DeclareTileOp>()) {
    if (!isDeclareTileReboundByTPop(decl)) {
      user->emitError("FoldTileBufIntrinsics: pto.declare_tile address requires "
                      "a matching pto.tpop rebinding");
      return std::nullopt;
    }
    auto tileTy = cast<pto::TileBufType>(decl.getResult().getType());
    return TileHandleInfo{decl.getResult(), Value{}, Value{},
                          tileTy.getConfigAttr()};
  }

  if (auto reshape = tileBuf.getDefiningOp<pto::TReshapeOp>()) {
    auto sourceInfo = resolveTileHandle(reshape.getSrc(), user);
    if (!sourceInfo) {
      return std::nullopt;
    }

    auto tileTy = dyn_cast<pto::TileBufType>(reshape.getResult().getType());
    if (!tileTy) {
      user->emitError(
          "FoldTileBufIntrinsics: pto.treshape must produce !pto.tile_buf");
      return std::nullopt;
    }

    auto [validRow, validCol] = findSetValidShapeOverride(tileBuf);
    return TileHandleInfo{sourceInfo->addr,
                          validRow ? validRow : sourceInfo->validRow,
                          validCol ? validCol : sourceInfo->validCol,
                          tileTy.getConfigAttr()};
  }

  user->emitError("FoldTileBufIntrinsics: expected tile_buf to be defined by "
                  "the active materialized tile-handle bridge "
                  "(pto.alloc_tile, pto.declare_tile rebound by pto.tpop, or "
                  "pto.treshape, "
                  "or a pto.fusion_region result that yields one of them)");
  return std::nullopt;
}

struct ViewChain {
  UnrealizedConversionCastOp cast;
  memref::SubViewOp subview;
  memref::ReinterpretCastOp reinterpretCast;
  Value baseMemref;
};

static std::optional<ViewChain> traceViewChain(Value tensorView,
                                               Operation *user) {
  Value memrefVal;
  UnrealizedConversionCastOp castOp;

  if (isa<MemRefType>(tensorView.getType())) {
    memrefVal = tensorView;
  } else {
    castOp = tensorView.getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp || castOp.getNumOperands() != 1) {
      user->emitError(
          "FoldTileBufIntrinsics: expected tensor_view to be defined by a "
          "single-operand builtin.unrealized_conversion_cast");
      return std::nullopt;
    }
    memrefVal = castOp.getOperand(0);
    if (!isa<MemRefType>(memrefVal.getType())) {
      user->emitError(
          "FoldTileBufIntrinsics: expected cast operand to be a memref, got ")
          << memrefVal.getType();
      return std::nullopt;
    }
  }

  if (auto memrefCastOp =
          memrefVal.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (memrefCastOp.getNumOperands() == 1 && memrefCastOp.getNumResults() == 1 &&
        isa<MemRefType>(memrefCastOp.getOperand(0).getType()) &&
        isa<MemRefType>(memrefCastOp.getResult(0).getType())) {
      castOp = memrefCastOp;
      memrefVal = memrefCastOp.getOperand(0);
    }
  }

  auto subviewOp = memrefVal.getDefiningOp<memref::SubViewOp>();
  if (!subviewOp) {
    user->emitError("FoldTileBufIntrinsics: expected memref to be defined by "
                    "memref.subview, got ")
        << (memrefVal.getDefiningOp()
                ? memrefVal.getDefiningOp()->getName().getStringRef()
                : StringRef("block argument"));
    return std::nullopt;
  }

  auto rcOp = subviewOp.getSource().getDefiningOp<memref::ReinterpretCastOp>();
  if (!rcOp) {
    user->emitError(
        "FoldTileBufIntrinsics: expected subview source to be defined by "
        "memref.reinterpret_cast, got ")
        << (subviewOp.getSource().getDefiningOp()
                ? subviewOp.getSource().getDefiningOp()->getName().getStringRef()
                : StringRef("block argument"));
    return std::nullopt;
  }

  return ViewChain{castOp, subviewOp, rcOp, rcOp.getSource()};
}

static bool getConstIndexValue(Value v, int64_t &out) {
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
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>()) {
    return getConstIndexValue(castOp.getIn(), out);
  }
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>()) {
    return getConstIndexValue(extOp.getIn(), out);
  }
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>()) {
    return getConstIndexValue(extOp.getIn(), out);
  }
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>()) {
    return getConstIndexValue(truncOp.getIn(), out);
  }
  return false;
}

static Value getValueOrCreateConstant(OpBuilder &builder, Location loc,
                                      OpFoldResult ofr) {
  if (auto val = dyn_cast<Value>(ofr)) {
    return val;
  }
  auto intAttr = dyn_cast<IntegerAttr>(cast<Attribute>(ofr));
  assert(intAttr && "expected integer attribute in OpFoldResult");
  return builder.create<arith::ConstantIndexOp>(loc, intAttr.getInt());
}

static bool isAllStaticZero(ArrayRef<OpFoldResult> ofrs) {
  for (OpFoldResult ofr : ofrs) {
    auto attr = dyn_cast<Attribute>(ofr);
    if (!attr) {
      return false;
    }
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (!intAttr || intAttr.getInt() != 0) {
      return false;
    }
  }
  return true;
}

static Value computeResultStride(OpBuilder &builder, Location loc,
                                 OpFoldResult rcStride,
                                 OpFoldResult svStride) {
  if (auto attr = dyn_cast<Attribute>(svStride)) {
    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (intAttr && intAttr.getInt() == 1) {
      return getValueOrCreateConstant(builder, loc, rcStride);
    }
  }

  Value lhs = getValueOrCreateConstant(builder, loc, rcStride);
  Value rhs = getValueOrCreateConstant(builder, loc, svStride);
  return builder.create<arith::MulIOp>(loc, lhs, rhs);
}

static Value computeLinearOffset(OpBuilder &builder, Location loc,
                                 ArrayRef<OpFoldResult> rcOffsets,
                                 ArrayRef<OpFoldResult> svOffsets,
                                 ArrayRef<OpFoldResult> rcStrides) {
  bool rcAllZero = isAllStaticZero(rcOffsets);
  bool svAllZero = isAllStaticZero(svOffsets);
  if (rcAllZero && svAllZero) {
    return Value();
  }

  Value svPart;
  if (!svAllZero) {
    for (auto [svOffset, rcStride] : llvm::zip(svOffsets, rcStrides)) {
      if (auto attr = dyn_cast<Attribute>(svOffset)) {
        auto intAttr = dyn_cast<IntegerAttr>(attr);
        if (intAttr && intAttr.getInt() == 0) {
          continue;
        }
      }

      Value off = getValueOrCreateConstant(builder, loc, svOffset);
      Value stride = getValueOrCreateConstant(builder, loc, rcStride);
      Value term = builder.create<arith::MulIOp>(loc, off, stride);
      svPart = svPart ? builder.create<arith::AddIOp>(loc, svPart, term) : term;
    }
  }

  Value rcPart;
  if (!rcAllZero) {
    if (rcOffsets.empty()) {
      return Value();
    }
    rcPart = getValueOrCreateConstant(builder, loc, rcOffsets.front());
  }

  if (rcPart && svPart) {
    return builder.create<arith::AddIOp>(loc, rcPart, svPart);
  }
  return rcPart ? rcPart : svPart;
}

static Value unwrapPTOViewBridge(Value value) {
  while (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getNumOperands() != 1 || cast.getNumResults() != 1) {
      break;
    }
    value = cast.getOperand(0);
  }
  return value;
}

enum class PTOViewProjectionKind { Dim, Stride, Address };

static Value projectSCFIfViewResult(Value view, PTOViewProjectionKind kind,
                                    int64_t dim, pto::PtrType resultPtrType,
                                    OpBuilder &builder, Operation *user);

static Value resolvePTOViewDim(Value view, int64_t dim, OpBuilder &builder,
                               Operation *user) {
  view = unwrapPTOViewBridge(view);
  if (Value projected = projectSCFIfViewResult(
          view, PTOViewProjectionKind::Dim, dim, {}, builder, user)) {
    return projected;
  }
  if (auto partition = view.getDefiningOp<pto::PartitionViewOp>()) {
    if (dim < 0 || dim >= static_cast<int64_t>(partition.getSizes().size())) {
      return {};
    }
    return partition.getSizes()[dim];
  }
  if (auto makeView = view.getDefiningOp<pto::MakeTensorViewOp>()) {
    if (dim < 0 || dim >= static_cast<int64_t>(makeView.getShape().size())) {
      return {};
    }
    return makeView.getShape()[dim];
  }
  (void)builder;
  (void)user;
  return {};
}

static Value resolvePTOViewStride(Value view, int64_t dim, OpBuilder &builder,
                                  Operation *user) {
  view = unwrapPTOViewBridge(view);
  if (Value projected = projectSCFIfViewResult(
          view, PTOViewProjectionKind::Stride, dim, {}, builder, user)) {
    return projected;
  }
  if (auto partition = view.getDefiningOp<pto::PartitionViewOp>()) {
    return resolvePTOViewStride(partition.getSource(), dim, builder, user);
  }
  if (auto makeView = view.getDefiningOp<pto::MakeTensorViewOp>()) {
    if (dim < 0 || dim >= static_cast<int64_t>(makeView.getStrides().size())) {
      return {};
    }
    return makeView.getStrides()[dim];
  }
  return {};
}

static Value resolvePTOViewAddress(Value view, pto::PtrType resultType,
                                   OpBuilder &builder, Operation *user) {
  view = unwrapPTOViewBridge(view);
  if (Value projected = projectSCFIfViewResult(
          view, PTOViewProjectionKind::Address, 0, resultType, builder, user)) {
    return projected;
  }
  if (auto makeView = view.getDefiningOp<pto::MakeTensorViewOp>()) {
    Value ptr = makeView.getPtr();
    if (ptr.getType() == resultType) {
      return ptr;
    }
    return builder.create<pto::CastPtrOp>(user->getLoc(), resultType, ptr);
  }
  auto partition = view.getDefiningOp<pto::PartitionViewOp>();
  if (!partition) {
    return {};
  }

  Value linearOffset;
  for (unsigned index = 0; index < partition.getOffsets().size(); ++index) {
    Value stride = resolvePTOViewStride(partition.getSource(), index, builder,
                                        user);
    if (!stride) {
      return {};
    }
    Value offset = partition.getOffsets()[index];
    Value term = builder.create<arith::MulIOp>(user->getLoc(), offset, stride);
    linearOffset = linearOffset
                       ? builder.create<arith::AddIOp>(user->getLoc(),
                                                       linearOffset, term)
                       : term;
  }
  Value base =
      resolvePTOViewAddress(partition.getSource(), resultType, builder, user);
  if (!base) {
    return {};
  }
  if (!linearOffset) {
    return base;
  }
  return builder.create<pto::AddPtrOp>(user->getLoc(), resultType, base,
                                       linearOffset);
}

static void cloneBlockWithoutTerminator(Block *source, Block *target,
                                        IRMapping &mapping,
                                        OpBuilder &builder) {
  builder.setInsertionPointToStart(target);
  for (Operation &op : source->without_terminator()) {
    builder.clone(op, mapping);
  }
}

static Value projectSCFIfViewResult(Value view, PTOViewProjectionKind kind,
                                    int64_t dim, pto::PtrType resultPtrType,
                                    OpBuilder &builder, Operation *user) {
  auto result = dyn_cast<OpResult>(view);
  auto ifOp = result ? dyn_cast<scf::IfOp>(result.getOwner()) : scf::IfOp();
  if (!ifOp || ifOp.getNumRegions() != mlir::pto::kValue2) {
    return {};
  }
  if (kind == PTOViewProjectionKind::Address && !resultPtrType) {
    return {};
  }
  auto thenYield = dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
  auto elseYield = dyn_cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
  unsigned resultIndex = result.getResultNumber();
  if (!thenYield || !elseYield ||
      resultIndex >= thenYield.getNumOperands() ||
      resultIndex >= elseYield.getNumOperands()) {
    return {};
  }

  Type projectionType = kind == PTOViewProjectionKind::Address
                            ? Type(resultPtrType)
                            : Type(builder.getIndexType());
  if (!projectionType) {
    return {};
  }
  SmallVector<Type> resultTypes(ifOp.getResultTypes().begin(),
                                ifOp.getResultTypes().end());
  resultTypes.push_back(projectionType);

  OpBuilder ifBuilder(ifOp);
  auto newIf = ifBuilder.create<scf::IfOp>(
      ifOp.getLoc(), resultTypes, ifOp.getCondition(),
      /*addThenBlock=*/true, /*addElseBlock=*/true);
  newIf->setAttrs(ifOp->getAttrs());

  auto cloneBranch = [&ifBuilder, dim, kind, resultIndex, resultPtrType, user](
                         Block *source, Block *target, scf::YieldOp oldYield,
                         bool &failed) {
    IRMapping mapping;
    cloneBlockWithoutTerminator(source, target, mapping, ifBuilder);
    SmallVector<Value> yields;
    yields.reserve(oldYield.getNumOperands() + 1);
    for (Value operand : oldYield.getOperands()) {
      yields.push_back(mapping.lookupOrDefault(operand));
    }
    Value yieldedView = yields[resultIndex];
    ifBuilder.setInsertionPointToEnd(target);
    Value projection;
    switch (kind) {
    case PTOViewProjectionKind::Dim:
      projection =
          resolvePTOViewDim(yieldedView, dim, ifBuilder, user);
      break;
    case PTOViewProjectionKind::Stride:
      projection =
          resolvePTOViewStride(yieldedView, dim, ifBuilder, user);
      break;
    case PTOViewProjectionKind::Address:
      projection = resolvePTOViewAddress(yieldedView, resultPtrType, ifBuilder,
                                         user);
      break;
    }
    if (!projection) {
      failed = true;
      return;
    }
    yields.push_back(projection);
    ifBuilder.create<scf::YieldOp>(oldYield.getLoc(), yields);
  };

  bool failed = false;
  cloneBranch(ifOp.thenBlock(), newIf.thenBlock(), thenYield, failed);
  cloneBranch(ifOp.elseBlock(), newIf.elseBlock(), elseYield, failed);
  if (failed) {
    newIf.erase();
    return {};
  }

  for (auto [oldResult, newResult] :
       llvm::zip_equal(ifOp.getResults(), newIf.getResults().take_front(
                                              ifOp.getNumResults()))) {
    oldResult.replaceAllUsesWith(newResult);
  }
  Value projection = newIf.getResult(newIf.getNumResults() - 1);
  ifOp.erase();
  return projection;
}

struct FoldTileBufIntrinsicsPass
    : public pto::impl::FoldTileBufIntrinsicsBase<FoldTileBufIntrinsicsPass> {
  using FoldTileBufIntrinsicsBase::FoldTileBufIntrinsicsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    FailureOr<FoldIntrinsicMode> mode = parseFoldIntrinsicMode(foldMode);
    if (failed(mode)) {
      func.emitError()
          << "FoldTileBufIntrinsics: unsupported --fold-mode value '"
          << foldMode << "' (expected all, shape-only, or addr-only)";
      return signalPassFailure();
    }

    // Leftover TileLang template instances (private, uncalled after
    // PTOInlineLibCall) still contain pto.tile_buf_addr / tile_valid_*
    // ops on tile_buf function arguments — they have no materialized tile
    // handle anchor to fold against and will be removed by later DCE. Skip
    // them.
    if (func->hasAttr("pto.tilelang.instance")) {
      return;
    }

    SmallVector<pto::TileBufAddrOp, mlir::pto::kValue8> addrOps;
    SmallVector<pto::TileValidRowsOp, mlir::pto::kValue8> rowsOps;
    SmallVector<pto::TileValidColsOp, mlir::pto::kValue8> colsOps;
    SmallVector<pto::TensorViewAddrOp, mlir::pto::kValue8> tvAddrOps;
    SmallVector<pto::GetTensorViewDimOp, mlir::pto::kValue8> tvDimOps;
    SmallVector<pto::GetTensorViewStrideOp, mlir::pto::kValue8> tvStrideOps;
    SmallVector<pto::GetValidShapeOp, mlir::pto::kValue8> getValidShapeOps;

    func.walk([&](Operation *op) {
      if (auto addr = dyn_cast<pto::TileBufAddrOp>(op)) {
        addrOps.push_back(addr);
      } else if (auto rows = dyn_cast<pto::TileValidRowsOp>(op)) {
        rowsOps.push_back(rows);
      } else if (auto cols = dyn_cast<pto::TileValidColsOp>(op)) {
        colsOps.push_back(cols);
      } else if (auto tvAddr = dyn_cast<pto::TensorViewAddrOp>(op)) {
        tvAddrOps.push_back(tvAddr);
      } else if (auto tvDim = dyn_cast<pto::GetTensorViewDimOp>(op)) {
        tvDimOps.push_back(tvDim);
      } else if (auto tvStride = dyn_cast<pto::GetTensorViewStrideOp>(op)) {
        tvStrideOps.push_back(tvStride);
      } else if (auto gvs = dyn_cast<pto::GetValidShapeOp>(op)) {
        getValidShapeOps.push_back(gvs);
      }
    });

    if (shouldFoldAddrFamily(*mode)) {
      // Fold pto.get_validshape into the materialized tile handle
      // valid_row / valid_col. This must precede tile_buf_addr and
      // tile_valid_{rows,cols} folding: set_validshape operands are usually
      // produced by get_validshape, so resolving them first lets
      // resolveTileHandle observe the overridden valid shape carried by a
      // treshape + set_validshape pair.
      for (auto gvsOp : getValidShapeOps) {
        if (gvsOp->hasAttr(kTileOpValidShapeReadAttr)) {
          continue;
        }
        if (!isa<pto::TileBufType>(gvsOp.getSource().getType())) {
          continue;
        }

        builder.setInsertionPoint(gvsOp);
        auto tileTy = cast<pto::TileBufType>(gvsOp.getSource().getType());
        auto validShape = tileTy.getValidShape();

        Value rowReplacement;
        if (!validShape.empty() && validShape[0] != ShapedType::kDynamic) {
          rowReplacement =
              builder.create<arith::ConstantIndexOp>(gvsOp.getLoc(), validShape[0]);
        }

        Value colReplacement;
        if (validShape.size() >= mlir::pto::kValue2 && validShape[1] != ShapedType::kDynamic) {
          colReplacement =
              builder.create<arith::ConstantIndexOp>(gvsOp.getLoc(), validShape[1]);
        }

        if (!rowReplacement || !colReplacement) {
          auto handleInfo = resolveTileHandle(gvsOp.getSource(), gvsOp);
          if (!handleInfo) {
            return signalPassFailure();
          }
          if (!rowReplacement) {
            rowReplacement = handleInfo->validRow;
          }
          if (!colReplacement) {
            colReplacement = handleInfo->validCol;
          }
        }

        if (!rowReplacement || !colReplacement) {
          gvsOp.emitError("FoldTileBufIntrinsics: pto.get_validshape could not "
                          "resolve a concrete valid_row / valid_col");
          return signalPassFailure();
        }

        gvsOp.getValidRow().replaceAllUsesWith(rowReplacement);
        gvsOp.getValidCol().replaceAllUsesWith(colReplacement);
        gvsOp.erase();
      }

      // Fold pto.tile_buf_addr by recovering the active materialized tile
      // handle contract:
      //   - pto.alloc_tile → cast the explicit addr to the requested pointer.
      // Memref sources are a legacy compatibility seam. They are already
      // materialized buffers, so keep them as identity markers or cast the
      // base memref to the requested pointer type without re-entering the
      // tile_buf handle path.
      for (auto addrOp : addrOps) {
        if (auto srcMemrefType =
                dyn_cast<MemRefType>(addrOp.getSrc().getType())) {
          if (auto resultMemrefType =
                  dyn_cast<MemRefType>(addrOp.getDst().getType())) {
            if (srcMemrefType != resultMemrefType) {
              addrOp.getDst().setType(srcMemrefType);
            }
            addrOp.getDst().replaceAllUsesWith(addrOp.getSrc());
            addrOp.erase();
            continue;
          }

          if (auto resultPtrType =
                  dyn_cast<pto::PtrType>(addrOp.getDst().getType())) {
            builder.setInsertionPoint(addrOp);
            Value replacement = builder.create<pto::CastPtrOp>(
                addrOp.getLoc(), resultPtrType, addrOp.getSrc());
            addrOp.getDst().replaceAllUsesWith(replacement);
            addrOp.erase();
            continue;
          }

          addrOp.emitError("FoldTileBufIntrinsics: tile_buf_addr result must "
                           "be memref or !pto.ptr");
          return signalPassFailure();
        }

        // An SCF result/iter_arg is already a runtime-selected tile handle.
        // Keep tile_buf_addr attached to that handle; VPTO pointer
        // normalization converts it directly without choosing one branch's
        // allocation address here.
        if (isSCFTileCarrier(addrOp.getSrc())) {
          continue;
        }

        // A declare_tile rebound by TPOP gets its address at runtime from the
        // FIFO slot; folding it to the placeholder value here would break the
        // VPTO pipe bridge address propagation.
        if (auto decl = addrOp.getSrc().getDefiningOp<pto::DeclareTileOp>();
            decl && isDeclareTileReboundByTPop(decl))
          continue;

        auto handleInfo = resolveTileHandle(addrOp.getSrc(), addrOp);
        if (!handleInfo) {
          return signalPassFailure();
        }

        auto tileTy = dyn_cast<pto::TileBufType>(addrOp.getSrc().getType());
        if (!tileTy) {
          addrOp.emitError("FoldTileBufIntrinsics: tile_buf_addr source must be "
                           "!pto.tile_buf");
          return signalPassFailure();
        }

        auto resultPtrType = dyn_cast<pto::PtrType>(addrOp.getDst().getType());
        if (!resultPtrType) {
          addrOp.emitError("FoldTileBufIntrinsics: tile_buf_addr result must "
                           "be !pto.ptr");
          return signalPassFailure();
        }

        if (!handleInfo->addr) {
          addrOp.emitError("FoldTileBufIntrinsics: pto.alloc_tile used by "
                           "tile_buf_addr must carry an addr operand on the "
                           "VPTO path");
          return signalPassFailure();
        }

        builder.setInsertionPoint(addrOp);
        Value replacement = builder.create<pto::CastPtrOp>(
            addrOp.getLoc(), resultPtrType, handleInfo->addr);
        addrOp.getDst().replaceAllUsesWith(replacement);
        addrOp.erase();
      }
    }

    if (shouldFoldShapeFamily(*mode)) {
      // Fold pto.tile_valid_rows → arith.constant (static) or the dynamic
      // valid_row operand carried by the new tile handle bridge.
      for (auto rowsOp : rowsOps) {
        builder.setInsertionPoint(rowsOp);
        auto tbTy = dyn_cast<pto::TileBufType>(rowsOp.getSrc().getType());
        if (!tbTy || tbTy.getValidShape().empty()) {
          rowsOp.emitError("tile_valid_rows: invalid tile_buf type");
          return signalPassFailure();
        }

        int64_t vRow = tbTy.getValidShape()[0];
        Value replacement;
        if (vRow != ShapedType::kDynamic) {
          replacement =
              builder.create<arith::ConstantIndexOp>(rowsOp.getLoc(), vRow);
        } else {
          auto handleInfo = resolveTileHandle(rowsOp.getSrc(), rowsOp);
          if (!handleInfo) {
            return signalPassFailure();
          }
          replacement = handleInfo->validRow;
          if (!replacement) {
            rowsOp.emitError(
                "tile_valid_rows: dynamic v_row but the materialized tile "
                "handle has no valid_row operand");
            return signalPassFailure();
          }
          assert(replacement.getType() == rowsOp.getResult().getType() &&
                 "tile_valid_rows fold: type mismatch with handle valid_row");
        }
        rowsOp.getResult().replaceAllUsesWith(replacement);
        rowsOp.erase();
      }

      // Fold pto.tile_valid_cols → arith.constant (static) or the dynamic
      // valid_col operand carried by the new tile handle bridge.
      for (auto colsOp : colsOps) {
        builder.setInsertionPoint(colsOp);
        auto tbTy = dyn_cast<pto::TileBufType>(colsOp.getSrc().getType());
        if (!tbTy || tbTy.getValidShape().size() < mlir::pto::kValue2) {
          colsOp.emitError("tile_valid_cols: invalid tile_buf type");
          return signalPassFailure();
        }

        int64_t vCol = tbTy.getValidShape()[1];
        Value replacement;
        if (vCol != ShapedType::kDynamic) {
          replacement =
              builder.create<arith::ConstantIndexOp>(colsOp.getLoc(), vCol);
        } else {
          auto handleInfo = resolveTileHandle(colsOp.getSrc(), colsOp);
          if (!handleInfo) {
            return signalPassFailure();
          }
          replacement = handleInfo->validCol;
          if (!replacement) {
            colsOp.emitError(
                "tile_valid_cols: dynamic v_col but the materialized tile "
                "handle has no valid_col operand");
            return signalPassFailure();
          }
          assert(replacement.getType() == colsOp.getResult().getType() &&
                 "tile_valid_cols fold: type mismatch with handle valid_col");
        }
        colsOp.getResult().replaceAllUsesWith(replacement);
        colsOp.erase();
      }

      for (auto dimOp : tvDimOps) {
        int64_t dimIdx = 0;
        if (!getConstIndexValue(dimOp.getDimIndex(), dimIdx)) {
          dimOp.emitError(
              "FoldTileBufIntrinsics: get_tensor_view_dim requires a constant "
              "dim index");
          return signalPassFailure();
        }

        builder.setInsertionPoint(dimOp);
        if (Value direct = resolvePTOViewDim(dimOp.getTensorView(), dimIdx,
                                             builder, dimOp)) {
          dimOp.getResult().replaceAllUsesWith(direct);
          dimOp.erase();
          continue;
        }

        auto chain = traceViewChain(dimOp.getTensorView(), dimOp);
        if (!chain) {
          return signalPassFailure();
        }

        auto svTy = cast<MemRefType>(chain->subview.getType());
        if (dimIdx < 0 || dimIdx >= svTy.getRank()) {
          dimOp.emitError(
              "FoldTileBufIntrinsics: get_tensor_view_dim dim index out of "
              "bounds");
          return signalPassFailure();
        }

        builder.setInsertionPoint(dimOp);
        Value replacement;
        if (!svTy.isDynamicDim(dimIdx)) {
          replacement =
              builder.create<arith::ConstantIndexOp>(dimOp.getLoc(),
                                                     svTy.getDimSize(dimIdx));
        } else {
          replacement = getValueOrCreateConstant(
              builder, dimOp.getLoc(), chain->subview.getMixedSizes()[dimIdx]);
        }

        dimOp.getResult().replaceAllUsesWith(replacement);
        dimOp.erase();
      }

      for (auto strideOp : tvStrideOps) {
        int64_t dimIdx = 0;
        if (!getConstIndexValue(strideOp.getDimIndex(), dimIdx)) {
          strideOp.emitError(
              "FoldTileBufIntrinsics: get_tensor_view_stride requires a "
              "constant dim index");
          return signalPassFailure();
        }

        builder.setInsertionPoint(strideOp);
        if (Value direct = resolvePTOViewStride(
                strideOp.getTensorView(), dimIdx, builder, strideOp)) {
          strideOp.getResult().replaceAllUsesWith(direct);
          strideOp.erase();
          continue;
        }

        auto chain = traceViewChain(strideOp.getTensorView(), strideOp);
        if (!chain) {
          return signalPassFailure();
        }

        auto svTy = cast<MemRefType>(chain->subview.getType());
        if (dimIdx < 0 || dimIdx >= svTy.getRank()) {
          strideOp.emitError(
              "FoldTileBufIntrinsics: get_tensor_view_stride dim index out of "
              "bounds");
          return signalPassFailure();
        }

        builder.setInsertionPoint(strideOp);
        Value replacement = computeResultStride(
            builder, strideOp.getLoc(),
            chain->reinterpretCast.getMixedStrides()[dimIdx],
            chain->subview.getMixedStrides()[dimIdx]);

        strideOp.getResult().replaceAllUsesWith(replacement);
        strideOp.erase();
      }
    }

    if (shouldFoldAddrFamily(*mode)) {
      for (auto addrOp : tvAddrOps) {
        builder.setInsertionPoint(addrOp);

        auto resultPtrType = dyn_cast<pto::PtrType>(addrOp.getDst().getType());
        if (resultPtrType) {
          if (Value direct = resolvePTOViewAddress(
                  addrOp.getSrc(), resultPtrType, builder, addrOp)) {
            addrOp.getDst().replaceAllUsesWith(direct);
            addrOp.erase();
            continue;
          }
        }

        auto chain = traceViewChain(addrOp.getSrc(), addrOp);
        if (!chain) {
          return signalPassFailure();
        }

        if (!resultPtrType) {
          if (auto resultMemrefType =
                  dyn_cast<MemRefType>(addrOp.getDst().getType())) {
            Value base = chain->baseMemref;
            if (base.getType() != resultMemrefType) {
              addrOp.getDst().setType(cast<MemRefType>(base.getType()));
            }
            addrOp.getDst().replaceAllUsesWith(base);
            addrOp.erase();
            continue;
          }
          addrOp.emitError(
              "FoldTileBufIntrinsics: tensor_view_addr result must be memref "
              "or !pto.ptr");
          return signalPassFailure();
        }

        Value linearOffset =
            computeLinearOffset(builder, addrOp.getLoc(),
                                chain->reinterpretCast.getMixedOffsets(),
                                chain->subview.getMixedOffsets(),
                                chain->reinterpretCast.getMixedStrides());

        Value basePtr = builder.create<pto::CastPtrOp>(
            addrOp.getLoc(), resultPtrType, chain->baseMemref);
        Value replacement =
            linearOffset
                ? builder.create<pto::AddPtrOp>(addrOp.getLoc(), resultPtrType,
                                                basePtr, linearOffset)
                : basePtr;

        addrOp.getDst().replaceAllUsesWith(replacement);
        addrOp.erase();
      }
    }

    // Clean up dead unrealized_conversion_cast ops that bridged
    // memref -> partition_tensor_view / tile_buf and are now unused
    // after folding.
    SmallVector<UnrealizedConversionCastOp, mlir::pto::kValue8> deadCasts;
    func.walk([&](UnrealizedConversionCastOp castOp) {
      if (castOp.use_empty() && castOp.getNumOperands() == 1 &&
          isa<MemRefType>(castOp.getOperand(0).getType()) &&
          isa<MemRefType, pto::PartitionTensorViewType, pto::TileBufType>(
              castOp.getResult(0).getType())) {
        deadCasts.push_back(castOp);
      }
    });
    for (auto castOp : llvm::reverse(deadCasts)) {
      castOp.erase();
    }

    while (true) {
      SmallVector<Operation *, mlir::pto::kValue8> deadMemrefOps;
      func.walk([&](Operation *op) {
        if ((isa<memref::SubViewOp>(op) ||
             isa<memref::ReinterpretCastOp>(op)) &&
            op->use_empty()) {
          deadMemrefOps.push_back(op);
        }
      });
      if (deadMemrefOps.empty()) {
        break;
      }
      for (auto *op : llvm::reverse(deadMemrefOps)) {
        op->erase();
      }
    }

    // Erase metadata writes only after every reader has been folded. TileOp
    // helper ABI reads marked above must remain runtime reads, so their
    // set_validshape updates are still observable by later lowering.
    SmallVector<pto::SetValidShapeOp, mlir::pto::kValue8> setValidShapeOps;
    func.walk([&](pto::SetValidShapeOp op) { setValidShapeOps.push_back(op); });
    for (auto op : llvm::reverse(setValidShapeOps)) {
      bool hasRuntimeReader = false;
      func.walk([&](pto::GetValidShapeOp reader) {
        if (reader.getSource() == op.getSource() &&
            reader->hasAttr(kTileOpValidShapeReadAttr)) {
          hasRuntimeReader = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!hasRuntimeReader) {
        op.erase();
      }
    }

    // DCE tile-handle view / alloc ops left behind after valid-shape
    // folding (treshape / alloc_tile / bridging casts).
    bool tileDceChanged = true;
    while (tileDceChanged) {
      tileDceChanged = false;
      SmallVector<Operation *, mlir::pto::kValue8> deadTileOps;
      func.walk([&](Operation *op) {
        if (!op->use_empty()) {
          return;
        }
        if (isa<pto::TReshapeOp, pto::AllocTileOp>(op)) {
          deadTileOps.push_back(op);
        }
        else if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
          if (castOp.getNumOperands() == 1 &&
              isa<pto::TileBufType>(castOp.getResult(0).getType())) {
            deadTileOps.push_back(op);
          }
        }
      });
      for (auto *op : llvm::reverse(deadTileOps)) {
        op->erase();
        tileDceChanged = true;
      }
    }

    eraseDeadViewChains(func);
  }
};

} // namespace

namespace mlir {
namespace pto {

std::unique_ptr<Pass> createFoldTileBufIntrinsicsPass() {
  return std::make_unique<FoldTileBufIntrinsicsPass>();
}

std::unique_ptr<Pass> createFoldTileBufIntrinsicsPass(llvm::StringRef foldMode) {
  FoldTileBufIntrinsicsOptions options;
  options.foldMode = foldMode.str();
  return std::make_unique<FoldTileBufIntrinsicsPass>(options);
}

} // namespace pto
} // namespace mlir

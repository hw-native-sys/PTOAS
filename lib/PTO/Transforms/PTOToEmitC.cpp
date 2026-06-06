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

static std::string getElemTypeStringForGT(Type elemTy);
static bool getStaticMemrefLayout(MemRefType mrTy,
                                  SmallVectorImpl<int64_t> &strides,
                                  int64_t &offset);
static int64_t multiplyOrDynamic(int64_t lhs, int64_t rhs);
static void buildGlobalTensorShapeAndStride(ArrayRef<int64_t> shape,
                                            ArrayRef<int64_t> strides,
                                            SmallVectorImpl<int64_t> &shape5D,
                                            SmallVectorImpl<int64_t> &stride5D);
static std::string joinIntTemplateParams(ArrayRef<int64_t> values);
static SmallVector<int64_t> buildRowMajorStrides(ArrayRef<int64_t> shape);
static std::string getGlobalTensorTypeStringFromShape(Type elemTy,
                                                      ArrayRef<int64_t> shape,
                                                      StringRef layoutEnum =
                                                          "pto::Layout::ND");
static std::string getGlobalTensorTypeStringFromShapeAndStrides(
    Type elemTy, ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
    StringRef layoutEnum = "pto::Layout::ND");
static emitc::OpaqueType getGlobalTensorOpaqueTypeFromShape(
    MLIRContext *ctx, Type elemTy, ArrayRef<int64_t> shape,
    StringRef layoutEnum = "pto::Layout::ND");

static const char *addrSpaceQualifier(pto::AddressSpace as) {
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
static constexpr llvm::StringLiteral kForceDynamicValidShapeAttrName =
    "__pto.force_dynamic_valid_shape";
static constexpr llvm::StringLiteral kGlobalTensorStridesAttrName =
    "__pto.globaltensor_strides";
static constexpr llvm::StringLiteral kTNotifyDrainMte2AttrName =
    "__pto.emitc.tnotify_drain_mte2";
static constexpr llvm::StringLiteral kTNotifyDrainMte3AttrName =
    "__pto.emitc.tnotify_drain_mte3";

enum TNotifyMteDrainMask : unsigned {
  kDrainMte2 = 1U << 0,
  kDrainMte3 = 1U << 1,
};
static constexpr llvm::StringLiteral kLastUseAttrName = "pto.last_use";
static constexpr llvm::StringLiteral kLastUseMarkerPrefix = "PTOAS__LAST_USE__";

static SmallVector<unsigned, 4> collectTileOperandNumbers(Operation *op) {
  SmallVector<unsigned, 4> tileOperandNumbers;
  for (OpOperand &operand : op->getOpOperands()) {
    if (isa<pto::TileBufType>(operand.get().getType()))
      tileOperandNumbers.push_back(operand.getOperandNumber());
  }
  return tileOperandNumbers;
}

static bool isDpsInitOperand(OpOperand &operand) {
  Operation *owner = operand.getOwner();
  if (auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(owner)) {
    for (OpOperand &init : dpsIface.getDpsInitsMutable()) {
      if (&init == &operand)
        return true;
    }
  }
  return false;
}

static SmallVector<unsigned, 4>
buildDefaultLastUseTileSlotOrder(Operation *op) {
  SmallVector<unsigned, 4> dpsInitTileOperands;
  SmallVector<unsigned, 4> nonDpsTileOperands;
  for (OpOperand &operand : op->getOpOperands()) {
    if (!isa<pto::TileBufType>(operand.get().getType()))
      continue;
    if (isDpsInitOperand(operand))
      dpsInitTileOperands.push_back(operand.getOperandNumber());
    else
      nonDpsTileOperands.push_back(operand.getOperandNumber());
  }

  // Most tile intrinsics lower as `CALLEE(dst, src0, src1, ...)`. When an op
  // has exactly one DPS init tile, treat that output slot as the leading
  // emitted tile operand so `[[pto::last_use(...)]]` aligns with the final
  // intrinsic call order.
  if (dpsInitTileOperands.size() == 1) {
    SmallVector<unsigned, 4> ordered{dpsInitTileOperands.front()};
    ordered.append(nonDpsTileOperands.begin(), nonDpsTileOperands.end());
    return ordered;
  }

  SmallVector<unsigned, 4> ordered = std::move(nonDpsTileOperands);
  ordered.append(dpsInitTileOperands.begin(), dpsInitTileOperands.end());
  return ordered;
}

static std::optional<std::string> buildLastUseMarkerCallee(Operation *op,
                                                           StringRef callee,
                                                           ArrayRef<unsigned> tileSlotOrder = {}) {
  auto lastUseAttr = dyn_cast_or_null<DenseI64ArrayAttr>(
      op->getAttr(kLastUseAttrName));
  if (!lastUseAttr)
    return std::nullopt;

  SmallVector<unsigned, 4> originalTileOperands = collectTileOperandNumbers(op);
  ArrayRef<int64_t> originalBits = lastUseAttr.asArrayRef();
  if (originalTileOperands.size() != originalBits.size())
    return std::nullopt;

  SmallVector<unsigned, 4> defaultTileSlotOrder;
  if (tileSlotOrder.empty()) {
    defaultTileSlotOrder = buildDefaultLastUseTileSlotOrder(op);
    tileSlotOrder = defaultTileSlotOrder;
  }
  if (tileSlotOrder.size() != originalBits.size())
    return std::nullopt;

  SmallVector<int64_t, 4> reorderedBits;
  reorderedBits.reserve(tileSlotOrder.size());
  for (unsigned operandNumber : tileSlotOrder) {
    bool found = false;
    for (auto [idx, originalOperandNumber] : llvm::enumerate(originalTileOperands)) {
      if (originalOperandNumber != operandNumber)
        continue;
      reorderedBits.push_back(originalBits[idx]);
      found = true;
      break;
    }
    if (!found)
      return std::nullopt;
  }

  std::string marker = kLastUseMarkerPrefix.str();
  marker.append(callee.str());
  marker.append("__");
  bool first = true;
  for (int64_t bit : reorderedBits) {
    if (!first)
      marker.append("__");
    first = false;
    marker.append(std::to_string(bit));
  }
  return marker;
}

static StringRef getLastUseAwareCallee(Operation *op, StringRef callee,
                                       std::string &storage,
                                       ArrayRef<unsigned> tileSlotOrder = {}) {
  std::optional<std::string> marker =
      buildLastUseMarkerCallee(op, callee, tileSlotOrder);
  if (!marker)
    return callee;
  storage = std::move(*marker);
  return storage;
}

static void createLastUseAwareOpaqueCall(ConversionPatternRewriter &rewriter,
                                         Operation *op, TypeRange resultTypes,
                                         StringRef callee,
                                         ValueRange operands,
                                         ArrayAttr args = ArrayAttr{},
                                         ArrayAttr templateArgs = ArrayAttr{},
                                         ArrayRef<unsigned> tileSlotOrder = {}) {
  std::string calleeStorage;
  StringRef effectiveCallee =
      getLastUseAwareCallee(op, callee, calleeStorage, tileSlotOrder);
  rewriter.create<emitc::CallOpaqueOp>(op->getLoc(), resultTypes,
                                       effectiveCallee, args, templateArgs,
                                       operands);
}

static Value peelUnrealized(Value v) {
  if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
    return castOp.getOperand(0);
  return v;
}

static unsigned getMteDrainMaskForPipe(pto::PIPE pipe) {
  switch (pipe) {
  case pto::PIPE::PIPE_MTE2:
    return kDrainMte2;
  case pto::PIPE::PIPE_MTE3:
    return kDrainMte3;
  case pto::PIPE::PIPE_ALL:
    return kDrainMte2 | kDrainMte3;
  default:
    return 0;
  }
}

static unsigned getDirectMteDrainMask(Operation *op) {
  if (auto pipeOp = dyn_cast<pto::OpPipeInterface>(op))
    return getMteDrainMaskForPipe(pipeOp.getPipe());
  return 0;
}

static unsigned collectMteDrainMask(Operation *op) {
  unsigned mask = getDirectMteDrainMask(op);
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Operation &nested : block)
        mask |= collectMteDrainMask(&nested);
  return mask;
}

static bool isLoopLikeOp(Operation *op) {
  return isa<scf::ForOp, scf::WhileOp, scf::ParallelOp, scf::ForallOp>(op);
}

static void setTNotifyDrainAttrs(pto::TNotifyOp op, unsigned mask) {
  op->removeAttr(kTNotifyDrainMte2AttrName);
  op->removeAttr(kTNotifyDrainMte3AttrName);
  if (mask & kDrainMte2)
    op->setAttr(kTNotifyDrainMte2AttrName, UnitAttr::get(op.getContext()));
  if (mask & kDrainMte3)
    op->setAttr(kTNotifyDrainMte3AttrName, UnitAttr::get(op.getContext()));
}

static void markNestedTNotifyWithMask(Operation *op, unsigned mask) {
  op->walk([&](pto::TNotifyOp notify) { setTNotifyDrainAttrs(notify, mask); });
}

static unsigned annotateTNotifyMteDrainForBlock(Block &block,
                                                unsigned entryPendingMask,
                                                unsigned loopCarriedMask) {
  unsigned pendingMask = entryPendingMask;
  for (Operation &op : block) {
    if (auto notify = dyn_cast<pto::TNotifyOp>(op)) {
      setTNotifyDrainAttrs(notify, pendingMask | loopCarriedMask);
      pendingMask = 0;
    }

    pendingMask |= getDirectMteDrainMask(&op);

    unsigned regionEntryMask = pendingMask;
    unsigned combinedRegionExitMask = 0;
    for (Region &region : op.getRegions()) {
      unsigned nestedLoopCarriedMask = loopCarriedMask;
      if (isLoopLikeOp(&op))
        nestedLoopCarriedMask |= collectMteDrainMask(&op);

      if (region.hasOneBlock()) {
        combinedRegionExitMask |= annotateTNotifyMteDrainForBlock(
            region.front(), regionEntryMask, nestedLoopCarriedMask);
      } else {
        unsigned regionMask = collectMteDrainMask(&op);
        markNestedTNotifyWithMask(&op, regionEntryMask | nestedLoopCarriedMask |
                                           regionMask);
        combinedRegionExitMask |= regionEntryMask | regionMask;
      }
    }
    pendingMask |= combinedRegionExitMask;

    if (auto barrier = dyn_cast<pto::BarrierOp>(op))
      pendingMask &= ~getMteDrainMaskForPipe(barrier.getPipe().getPipe());
  }
  return pendingMask;
}

static void annotateTNotifyMteDrain(ModuleOp module) {
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.getBody().hasOneBlock()) {
      (void)annotateTNotifyMteDrainForBlock(func.getBody().front(),
                                            /*entryPendingMask=*/0,
                                            /*loopCarriedMask=*/0);
      continue;
    }

    // Be conservative for pre-existing CFG: without a path-sensitive CFG data
    // flow here, every TNotify may observe any MTE work in the function.
    unsigned funcMask = collectMteDrainMask(func.getOperation());
    markNestedTNotifyWithMask(func.getOperation(), funcMask);
  }
}

static Value buildGlobalTensorFromMemref(ConversionPatternRewriter &rewriter,
                                         Location loc, Value basePtr,
                                         MemRefType mrTy, Operation *anchor);

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
  SmallVector<int64_t, 2> shape;
  bool rowMajor = false;
  bool colMajor = false;
};

static std::optional<GatherScatterShapeLayoutInfo>
getGatherScatterShapeLayoutInfo(Type ty) {
  if (auto tileTy = dyn_cast<pto::TileBufType>(ty)) {
    ArrayRef<int64_t> validShape = tileTy.getValidShape();
    if (validShape.size() != 2)
      return std::nullopt;

    GatherScatterShapeLayoutInfo info;
    info.shape.assign(validShape.begin(), validShape.end());
    int32_t blayout = tileTy.getBLayoutValueI32();
    info.rowMajor = blayout == static_cast<int32_t>(pto::BLayout::RowMajor);
    info.colMajor = blayout == static_cast<int32_t>(pto::BLayout::ColMajor);
    return info;
  }

  auto memRefTy = dyn_cast<MemRefType>(ty);
  if (!memRefTy || memRefTy.getRank() != 2)
    return std::nullopt;

  SmallVector<int64_t, 4> strides;
  int64_t offset = ShapedType::kDynamic;
  if (failed(getStridesAndOffset(memRefTy, strides, offset)) ||
      strides.size() != 2)
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

static std::optional<mlir::pto::Layout>
resolveLayoutForGlobalTensor(Operation *anchor, Value basePtr) {
  if (auto layout = getLayoutAttrFromOp(anchor))
    return layout;
  return resolveLayoutFromValueChain(basePtr);
}

static std::string layoutToEmitCString(mlir::pto::Layout layout) {
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

static bool isEmitCGlobalTensorLikeType(Type ty) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty);
  return opaqueTy && opaqueTy.getValue().contains("GlobalTensor<");
}

static std::string getEmitCScalarTypeToken(Type elemTy) {
  if (pto::isPTOFloat8Type(elemTy) &&
      (elemTy.isFloat8E4M3() || elemTy.isFloat8E4M3FN() ||
       elemTy.isFloat8E4M3FNUZ() || elemTy.isFloat8E4M3B11FNUZ()))
    return "float8_e4m3_t";
  if (pto::isPTOFloat8Type(elemTy) &&
      (elemTy.isFloat8E5M2() || elemTy.isFloat8E5M2FNUZ()))
    return "float8_e5m2_t";
  if (isa<pto::HiF8Type>(elemTy))
    return "hifloat8_t";
  if (isa<pto::F4E1M2x2Type>(elemTy))
    return "float4_e1m2x2_t";
  if (isa<pto::F4E2M1x2Type>(elemTy))
    return "float4_e2m1x2_t";
  if (elemTy.isF16())
    return "half";
  if (elemTy.isBF16())
    return "bfloat16_t";
  if (elemTy.isF32())
    return "float";
  if (elemTy.isF64())
    return "double";
  if (elemTy.isInteger(8))
    return (elemTy.isSignlessInteger(8) || elemTy.isSignedInteger(8)) ? "int8_t"
                                                                       : "uint8_t";
  if (elemTy.isInteger(16))
    return (elemTy.isSignlessInteger(16) || elemTy.isSignedInteger(16))
               ? "int16_t"
               : "uint16_t";
  if (elemTy.isInteger(32))
    return (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
               ? "int32_t"
               : "uint32_t";
  if (elemTy.isInteger(64))
    return cast<IntegerType>(elemTy).isUnsigned() ? "uint64_t" : "int64_t";
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

static int64_t getEmitCScalarByteWidth(Type elemTy) {
  if (pto::getPTOStorageElemByteSize(elemTy) == 1)
    return 1;
  if (elemTy.isF16() || elemTy.isBF16() || elemTy.isInteger(16))
    return 2;
  if (elemTy.isF32() || elemTy.isInteger(32))
    return 4;
  if (elemTy.isF64() || elemTy.isInteger(64))
    return 8;
  return 4;
}

static std::string tileBufBLayoutToken(pto::TileBufConfigAttr configAttr);
static std::string tileBufSLayoutToken(pto::TileBufConfigAttr configAttr);
static std::string tileBufPadToken(pto::TileBufConfigAttr configAttr);
static pto::BLayout getTileBufBLayoutValue(pto::TileBufConfigAttr configAttr);
static int64_t renderTileTemplateDim(int64_t rawDim, Type elemTy,
                                     pto::BLayout blayout, int dimIdx);

static const char *tileRoleToken(Attribute memorySpace) {
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
      return "TileType::Vec";
    }
  }
  return "TileType::Vec";
}

static std::string tileBufCompactToken(pto::TileBufConfigAttr configAttr) {
  std::string compactTok = "CompactMode::Null";
  if (auto compactAttr = dyn_cast<CompactModeAttr>(configAttr.getCompactMode())) {
    switch (static_cast<int32_t>(compactAttr.getValue())) {
    case 1:
      compactTok = "CompactMode::Normal";
      break;
    case 2:
      compactTok = "CompactMode::RowPlusOne";
      break;
    default:
      compactTok = "CompactMode::Null";
      break;
    }
  }
  return compactTok;
}

static std::optional<std::string> getEmitCTileTypeString(pto::TileBufType type) {
  if (type.getRank() != 2)
    return std::nullopt;
  auto validShape = type.getValidShape();
  if (validShape.size() != 2)
    return std::nullopt;

  Type elemTy = type.getElementType();
  auto configAttr = type.getConfigAttr();
  pto::BLayout blayout = getTileBufBLayoutValue(configAttr);
  ArrayRef<int64_t> shape = type.getShape();
  int64_t rows = shape[0];
  int64_t cols = shape[1];

  auto render = [&](int64_t dim, int dimIdx) {
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

  int32_t fractal = 512;
  if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
    fractal = frAttr.getInt();

  return std::string("Tile<") + tileRoleToken(type.getMemorySpace()) + ", " +
         getEmitCScalarTypeToken(elemTy) + ", " +
         std::to_string(render(rows, 0)) + ", " +
         std::to_string(render(cols, 1)) + ", " +
         tileBufBLayoutToken(configAttr) + ", " + vrowTok + ", " + vcolTok +
         ", " + tileBufSLayoutToken(configAttr) + ", " +
         std::to_string(fractal) + ", " + tileBufPadToken(configAttr) + ", " +
         tileBufCompactToken(configAttr) + ">";
}

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class PTOToEmitCTypeConverter : public TypeConverter {
public:
  PTOToEmitCTypeConverter(MLIRContext *Ctx, PTOArch targetArch) {
    // ---------------------------------------------------------
    // 1. 基本类型 (f32, i32, index)
    // ---------------------------------------------------------
    addConversion([Ctx](FloatType type) -> Type {
      if (type.isFloat8E4M3() || type.isFloat8E4M3FN() ||
          type.isFloat8E4M3FNUZ() || type.isFloat8E4M3B11FNUZ())
        return emitc::OpaqueType::get(Ctx, "float8_e4m3_t");
      if (type.isFloat8E5M2() || type.isFloat8E5M2FNUZ())
        return emitc::OpaqueType::get(Ctx, "float8_e5m2_t");
      if (type.isF32()) return emitc::OpaqueType::get(Ctx, "float");
      if (type.isF16()) return emitc::OpaqueType::get(Ctx, "half");
      if (type.isBF16()) return emitc::OpaqueType::get(Ctx, "bfloat16_t");
      if (type.isF64()) return emitc::OpaqueType::get(Ctx, "double");
      llvm::errs() << "[Debug] Unsupported FloatType: " << type << "\n";
      return Type{};
    });

    addConversion([Ctx](pto::HiF8Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "hifloat8_t");
    });
    addConversion([Ctx](pto::F4E1M2x2Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "float4_e1m2x2_t");
    });
    addConversion([Ctx](pto::F4E2M1x2Type) -> Type {
      return emitc::OpaqueType::get(Ctx, "float4_e2m1x2_t");
    });

    addConversion([Ctx](IntegerType type) -> Type {
      if (type.getWidth() == 1)
        return type;

      // Prefer fixed-width C types. Preserve signedness if the MLIR integer is
      // explicitly signed/unsigned; treat signless as signed by default.
      const bool isUnsigned = type.isUnsignedInteger();
      switch (type.getWidth()) {
      case 8:
        return emitc::OpaqueType::get(Ctx, isUnsigned ? "uint8_t" : "int8_t");
      case 16:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint16_t" : "int16_t");
      case 32:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint32_t" : "int32_t");
      case 64:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint64_t" : "int64_t");
      default:
        llvm::errs() << "[Debug] Unsupported IntegerType width: "
                     << type.getWidth() << "\n";
        return emitc::OpaqueType::get(Ctx, "int32_t"); // Fallback
      }
    });

    addConversion([Ctx](IndexType type) -> Type {
      return emitc::OpaqueType::get(Ctx, "int32_t");
    });

    // vector<4xi16> (e.g. TMRGSORT executedNumList) -> pto::MrgSortExecutedNumList
    addConversion([Ctx](VectorType type) -> Type {
      if (type.getRank() == 1 && type.getNumElements() == 4 &&
          type.getElementType().isInteger(16))
        return emitc::OpaqueType::get(Ctx, "pto::MrgSortExecutedNumList");
      return Type{};
    });

    // ---------------------------------------------------------
    // 2. PTO 特殊类型 (透传或转换)
    // ---------------------------------------------------------
    addConversion([](emitc::OpaqueType type) { return type; });
    addConversion([](emitc::PointerType type) { return type; });

    // ---------------------------------------------------------
    // 2.5 PtrType 转换 (指针类型)
    // ---------------------------------------------------------
    addConversion([this, Ctx](pto::PtrType type) -> std::optional<Type> {
      Type elemType = type.getElementType();
      Type newElemType = convertType(elemType);
      if (!newElemType)
        return std::nullopt;

      std::string elemTypeStr;
      if (auto opq = dyn_cast<emitc::OpaqueType>(newElemType)) {
        elemTypeStr = opq.getValue().str();
      } else {
        llvm::errs() << "  [Error] PtrType elem type is not OpaqueType: "
                     << newElemType << "\n";
        return std::nullopt;
      }

      std::string qualifier = "__gm__";

      std::string finalTypeStr = qualifier + " " + elemTypeStr;
      return getEmitCPointerType(Ctx, finalTypeStr);
    });

    addConversion([Ctx](pto::PipeType type) -> Type {
      (void)type;
      return emitc::OpaqueType::get(Ctx, "auto");
    });

    addConversion([Ctx](pto::EventIdArrayType type) -> Type {
      std::string tok = "PTOAS_EventIdArray<" + std::to_string(type.getSize()) + ">";
      return emitc::OpaqueType::get(Ctx, tok);
    });

    // !pto.local_array<D1 x D2 x ... x T> -> !emitc.array<D1 x D2 x ... x T>.
    // Variables of this type render as `T a[D1][D2]...;` in the emitted C++.
    addConversion([this](pto::LocalArrayType type) -> std::optional<Type> {
      Type convertedElem = convertType(type.getElementType());
      if (!convertedElem)
        return std::nullopt;
      return emitc::ArrayType::get(type.getShape(), convertedElem);
    });

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
      return getGlobalTensorOpaqueTypeFromShape(
          Ctx, type.getElementType(), type.getShape());
    });

    addConversion([Ctx](pto::PartitionTensorViewType type) -> Type {
      return getGlobalTensorOpaqueTypeFromShape(
          Ctx, type.getElementType(), type.getShape());
    });

    addConversion([Ctx](pto::TileBufType type) -> std::optional<Type> {
      auto typeString = getEmitCTileTypeString(type);
      if (!typeString)
        return std::nullopt;
      return emitc::OpaqueType::get(Ctx, *typeString);
    });

    // ---------------------------------------------------------
    // 3. MemRef 转换 (Debug 重点)
    // ---------------------------------------------------------
    addConversion([this, Ctx](MemRefType type) -> std::optional<Type> {
      LLVM_DEBUG(llvm::dbgs() << "Converting MemRef: " << type << "\n");

      // A. 转换元素类型
      Type elemType = type.getElementType();
      Type newElemType = convertType(elemType); 
      if (!newElemType) {
        llvm::errs() << "  [Error] Failed to convert element type: " << elemType << "\n";
        return std::nullopt;
      }
      
      // 获取元素类型的字符串
      std::string elemTypeStr;
      if (auto opq = dyn_cast<emitc::OpaqueType>(newElemType)) {
        elemTypeStr = opq.getValue().str();
      } else {
         llvm::errs() << "  [Error] Converted element type is not OpaqueType: " << newElemType << "\n";
         return std::nullopt;
      }

      // B. 处理 Memory Space
      std::string qualifier = "";
      Attribute memorySpace = type.getMemorySpace();
      
      if (!memorySpace) {
         qualifier = "__gm__";
      } else if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(memorySpace)) {
         qualifier = addrSpaceQualifier(ptoAttr.getAddressSpace());
      } else {
         llvm::errs() << "  [Warning] Unknown MemorySpace Attribute type: " << memorySpace << "\n";
         qualifier = "__gm__"; // Fallback
      }

      std::string finalTypeStr = qualifier + " " + elemTypeStr;
      LLVM_DEBUG(llvm::dbgs() << "  [Success] -> " << finalTypeStr << "*\n");
      
      return getEmitCPointerType(Ctx, finalTypeStr);
    });

    // ---------------------------------------------------------
    // 4. Function & Materialization
    // ---------------------------------------------------------
    addConversion([this](FunctionType type) -> Type {
      SmallVector<Type> inputs;
      if (failed(convertTypes(type.getInputs(), inputs))) return Type{};
      SmallVector<Type> results;
      if (failed(convertTypes(type.getResults(), results))) return Type{};
      return FunctionType::get(type.getContext(), inputs, results);
    });

    auto materializeCast = [](OpBuilder &Builder, Type ResultType,
                              ValueRange Inputs, Location Loc) -> Value {
      if (Inputs.size() != 1) return Value();
      return Builder.create<UnrealizedConversionCastOp>(Loc, ResultType, Inputs[0]).getResult(0);
    };

    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
    // Needed for region/block signature conversions (e.g. CFG block args).
    addArgumentMaterialization(materializeCast);
  }
};

static constexpr unsigned kPTOIndexBitWidth =
    32; // keep consistent with IndexType conversion

// Forward declarations (definitions below).
static inline std::string pipeTokFromPipeAttr(mlir::pto::PipeAttr a);
static emitc::OpaqueType getSignedIntOpaqueType(MLIRContext *ctx,
                                                unsigned bitWidth);
static emitc::OpaqueType getUnsignedIntOpaqueType(MLIRContext *ctx,
                                                  unsigned bitWidth);
static emitc::OpaqueType getWiderSignedIntOpaqueType(MLIRContext *ctx,
                                                     unsigned bitWidth);
static emitc::OpaqueType getWiderUnsignedIntOpaqueType(MLIRContext *ctx,
                                                       unsigned bitWidth);
static Value makeEmitCOpaqueConstant(ConversionPatternRewriter &rewriter,
                                     Location loc, Type type,
                                     llvm::StringRef literal);
static Value makeEmitCIntConstant(ConversionPatternRewriter &rewriter,
                                  Location loc, Type type, int64_t value);
static Value emitCCast(ConversionPatternRewriter &rewriter, Location loc,
                       Type dstType, Value src);
static FailureOr<std::string> buildEmitCOpaqueConstantLiteral(Type targetType,
                                                              Attribute valueAttr);
static Value castSignlessIntToUnsignedSameWidth(ConversionPatternRewriter &rewriter,
                                                Location loc, Value v,
                                                unsigned bitWidth);
static bool needsA5NoSplitVectorGuard(Operation *op);

static FailureOr<std::string> getTileSplitToken(int64_t split) {
  switch (split) {
  case 0:
    return std::string("TileSplitAxis::TILE_NO_SPLIT");
  case 1:
    return std::string("TileSplitAxis::TILE_UP_DOWN");
  case 2:
    return std::string("TileSplitAxis::TILE_LEFT_RIGHT");
  default:
    return failure();
  }
}

static FailureOr<std::string>
getTPipeDirectionToken(bool isL2G2L, int8_t dirMask, PTOArch targetArch) {
  if (dirMask == 1) {
    if (isL2G2L && targetArch == PTOArch::A5)
      return std::string("Direction::DIR_C2V_GM");
    return std::string("Direction::DIR_C2V");
  }
  if (dirMask == 2) {
    if (isL2G2L && targetArch == PTOArch::A5)
      return std::string("Direction::DIR_V2C_GM");
    return std::string("Direction::DIR_V2C");
  }
  if (dirMask == 3)
    return std::string("Direction::DIR_BOTH");
  return failure();
}

static std::string buildTPipeToken(int32_t flagBase, llvm::StringRef dirTok,
                                   int32_t slotSize, int32_t slotNum,
                                   int32_t localSlotNum, bool nosplit) {
  std::string token = "TPipe<" + std::to_string(flagBase) + ", " + dirTok.str() +
                      ", " + std::to_string(slotSize) + ", " +
                      std::to_string(slotNum);
  token += ", " + std::to_string(localSlotNum);
  token += nosplit ? ", true" : ", false";
  token += ">";
  return token;
}

static FailureOr<std::string> buildTPipeTokenFromInitOp(Operation *op,
                                                        PTOArch targetArch) {
  if (auto initOp = dyn_cast<pto::InitializeL2G2LPipeOp>(op)) {
    if (!initOp.getFlagBaseAttr())
      return failure();
    auto dirTok =
        getTPipeDirectionToken(/*isL2G2L=*/true, initOp.getDirMask(), targetArch);
    if (failed(dirTok))
      return failure();
    int32_t localSlotNum = initOp.getLocalSlotNumAttr()
                               ? initOp.getLocalSlotNumAttr().getInt()
                               : initOp.getSlotNum();
    return buildTPipeToken(initOp.getFlagBaseAttr().getInt(), *dirTok,
                           initOp.getSlotSize(), initOp.getSlotNum(),
                           localSlotNum,
                           initOp.getNosplitAttr() &&
                               initOp.getNosplitAttr().getValue());
  }

  if (auto initOp = dyn_cast<pto::InitializeL2LPipeOp>(op)) {
    if (!initOp.getFlagBaseAttr())
      return failure();
    auto dirTok =
        getTPipeDirectionToken(/*isL2G2L=*/false, initOp.getDirMask(), targetArch);
    if (failed(dirTok))
      return failure();
    return buildTPipeToken(initOp.getFlagBaseAttr().getInt(), *dirTok,
                           initOp.getSlotSize(), initOp.getSlotNum(), 2,
                           initOp.getNosplitAttr() &&
                               initOp.getNosplitAttr().getValue());
  }

  return failure();
}

static FailureOr<std::string> getTPipeTokenFromValue(Value pipeHandle,
                                                     PTOArch targetArch) {
  pipeHandle = peelUnrealized(pipeHandle);
  Operation *def = pipeHandle.getDefiningOp();
  if (!def)
    return failure();
  return buildTPipeTokenFromInitOp(def, targetArch);
}

static bool isSetFFTsPointerLikeType(Type ty) {
  return isEmitCPointerLikeType(ty);
}

static bool tileDataReturnsIntegralAddress(pto::AddressSpace as) {
  return as == pto::AddressSpace::BIAS;
}

static Type getTileDataResultType(MLIRContext *ctx, pto::AddressSpace as,
                                  StringRef elemTok) {
  if (tileDataReturnsIntegralAddress(as))
    return emitc::OpaqueType::get(ctx, "uint64_t");
  return getEmitCPointerType(ctx, addrSpaceQualifier(as), elemTok);
}

static Value materializeTileDataValue(ConversionPatternRewriter &rewriter,
                                      Location loc, Value tile,
                                      pto::AddressSpace as,
                                      StringRef elemTok) {
  auto rawTy = getTileDataResultType(rewriter.getContext(), as, elemTok);
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, rawTy, "PTOAS__TILE_DATA",
                                   ArrayAttr{}, ArrayAttr{},
                                   ValueRange{tile})
      .getResult(0);
}

static Value materializeAddressAsPointer(ConversionPatternRewriter &rewriter,
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

struct InterCoreSyncCallDesc {
  const char *callee = nullptr;
  ArrayAttr args;
  SmallVector<Value, 2> operands;
};

static Value castInterCoreEventIdToI32(ConversionPatternRewriter &rewriter,
                                       Location loc, Value eventId) {
  auto i32Ty = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
  if (eventId.getType() == i32Ty)
    return eventId;
  return emitCCast(rewriter, loc, i32Ty, eventId);
}

static Attribute getFFTSModeCodegenArg(ConversionPatternRewriter &rewriter,
                                       int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  if (fftsMode == 2)
    return emitc::OpaqueAttr::get(ctx, "FFTS_MODE_VAL");
  return emitc::OpaqueAttr::get(ctx, std::to_string(fftsMode));
}

static Value createFFTSMsg(ConversionPatternRewriter &rewriter, Location loc,
                           Value eventI32, int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  auto msgTy = emitc::OpaqueType::get(ctx, "uint16_t");
  auto msgArgs = rewriter.getArrayAttr({
      getFFTSModeCodegenArg(rewriter, fftsMode),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, msgTy, "getFFTSMsg",
                                   /*args=*/msgArgs,
                                   /*templateArgs=*/ArrayAttr{},
                                   /*operands=*/ValueRange{eventI32})
      .getResult(0);
}

static InterCoreSyncCallDesc buildInterCoreSyncSetCall(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, IntegerAttr eventIdAttr, int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);

  if (targetArch == PTOArch::A3) {
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value eventVal =
        makeEmitCIntConstant(rewriter, loc, i32Ty, eventIdAttr.getInt());
    Value msgVal = createFFTSMsg(rewriter, loc, eventVal, fftsMode);

    InterCoreSyncCallDesc desc;
    desc.callee = "ffts_cross_core_sync";
    desc.args = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        IntegerAttr::get(IndexType::get(ctx), 0),
    });
    desc.operands.push_back(msgVal);
    return desc;
  }

  InterCoreSyncCallDesc desc;
  desc.callee = "set_intra_block";
  desc.args = rewriter.getArrayAttr(
      {emitc::OpaqueAttr::get(ctx, pipeTok), eventIdAttr});
  return desc;
}

static InterCoreSyncCallDesc buildInterCoreSyncSetCallDyn(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, Value eventIdVal, int64_t fftsMode) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);
  Value eventI32 = castInterCoreEventIdToI32(rewriter, loc, eventIdVal);

  if (targetArch == PTOArch::A3) {
    Value msgVal = createFFTSMsg(rewriter, loc, eventI32, fftsMode);

    InterCoreSyncCallDesc desc;
    desc.callee = "ffts_cross_core_sync";
    desc.args = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        IntegerAttr::get(IndexType::get(ctx), 0),
    });
    desc.operands.push_back(msgVal);
    return desc;
  }

  InterCoreSyncCallDesc desc;
  desc.callee = "set_intra_block";
  desc.args = rewriter.getArrayAttr({
      emitc::OpaqueAttr::get(ctx, pipeTok),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
  desc.operands.push_back(eventI32);
  return desc;
}

static InterCoreSyncCallDesc buildInterCoreSyncWaitCall(
    ConversionPatternRewriter &rewriter, PTOArch targetArch,
    pto::PipeAttr pipeAttr, IntegerAttr eventIdAttr) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);

  InterCoreSyncCallDesc desc;
  if (targetArch == PTOArch::A3) {
    desc.callee = "wait_flag_dev";
    desc.args = rewriter.getArrayAttr({eventIdAttr});
    return desc;
  }

  desc.callee = "wait_intra_block";
  desc.args = rewriter.getArrayAttr(
      {emitc::OpaqueAttr::get(ctx, pipeTok), eventIdAttr});
  return desc;
}

static InterCoreSyncCallDesc buildInterCoreSyncWaitCallDyn(
    ConversionPatternRewriter &rewriter, Location loc, PTOArch targetArch,
    pto::PipeAttr pipeAttr, Value eventIdVal) {
  auto *ctx = rewriter.getContext();
  std::string pipeTok = pipeTokFromPipeAttr(pipeAttr);
  Value eventI32 = castInterCoreEventIdToI32(rewriter, loc, eventIdVal);

  InterCoreSyncCallDesc desc;
  if (targetArch == PTOArch::A3) {
    desc.callee = "wait_flag_dev";
    desc.args = rewriter.getArrayAttr({IntegerAttr::get(IndexType::get(ctx), 0)});
    desc.operands.push_back(eventI32);
    return desc;
  }

  desc.callee = "wait_intra_block";
  desc.args = rewriter.getArrayAttr({
      emitc::OpaqueAttr::get(ctx, pipeTok),
      IntegerAttr::get(IndexType::get(ctx), 0),
  });
  desc.operands.push_back(eventI32);
  return desc;
}

static bool hasInterCoreSyncOp(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
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
  func.walk([&](Operation *op) {
    if (isa<pto::SetFFTsOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

//===----------------------------------------------------------------------===//
// Arith -> EmitC (full dialect coverage for scalar ops)
//===----------------------------------------------------------------------===//

template <typename ArithOp, typename EmitCOp>
struct ArithSimpleBinaryToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<EmitCOp>(op, dstTy, adaptor.getOperands());
    return success();
  }
};

// Integer bitwise ops (andi/ori/xori) on signless integers: perform in unsigned
// to avoid signedness pitfalls, then cast back.
template <typename ArithOp, typename EmitCOp>
struct ArithUnsignedBitwiseBinaryToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = this->getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    if (bitWidth == 1) {
      rewriter.replaceOpWithNewOp<EmitCOp>(op, dstTy, adaptor.getLhs(),
                                           adaptor.getRhs());
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value resU = rewriter.create<EmitCOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, resU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithDivUIToEmitC : public OpConversionPattern<arith::DivUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::DivUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value divU = rewriter.create<emitc::DivOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, divU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithRemUIToEmitC : public OpConversionPattern<arith::RemUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::RemUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value remU = rewriter.create<emitc::RemOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, remU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithCeilDivUIToEmitC : public OpConversionPattern<arith::CeilDivUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::CeilDivUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value one = makeEmitCIntConstant(rewriter, loc, uTy, 1);
    Value rhsMinusOne = rewriter.create<emitc::SubOp>(loc, uTy, rhsU, one);
    Value num = rewriter.create<emitc::AddOp>(loc, uTy, lhsU, rhsMinusOne);
    Value divU = rewriter.create<emitc::DivOp>(loc, uTy, num, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, divU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithCeilDivSIToEmitC : public OpConversionPattern<arith::CeilDivSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::CeilDivSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    Value zero = makeEmitCIntConstant(rewriter, loc, dstTy, 0);
    Value one = makeEmitCIntConstant(rewriter, loc, dstTy, 1);

    Value q0 = rewriter.create<emitc::DivOp>(loc, dstTy, adaptor.getLhs(),
                                             adaptor.getRhs());
    Value r = rewriter.create<emitc::RemOp>(loc, dstTy, adaptor.getLhs(),
                                            adaptor.getRhs());

    Value rNeZero = rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                                  emitc::CmpPredicate::ne, r,
                                                  zero);
    Value lhsLt0 =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::lt, adaptor.getLhs(),
                                      zero);
    Value rhsLt0 =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::lt, adaptor.getRhs(),
                                      zero);
    Value signsSame =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::eq, lhsLt0, rhsLt0);
    Value adjust =
        rewriter.create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
                                             rNeZero, signsSame);

    Value qPlusOne = rewriter.create<emitc::AddOp>(loc, dstTy, q0, one);
    Value result = rewriter.create<emitc::ConditionalOp>(loc, dstTy, adjust,
                                                         qPlusOne, q0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithFloorDivSIToEmitC : public OpConversionPattern<arith::FloorDivSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::FloorDivSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    Value zero = makeEmitCIntConstant(rewriter, loc, dstTy, 0);
    Value one = makeEmitCIntConstant(rewriter, loc, dstTy, 1);

    Value q0 = rewriter.create<emitc::DivOp>(loc, dstTy, adaptor.getLhs(),
                                             adaptor.getRhs());
    Value r = rewriter.create<emitc::RemOp>(loc, dstTy, adaptor.getLhs(),
                                            adaptor.getRhs());

    Value rNeZero = rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                                  emitc::CmpPredicate::ne, r,
                                                  zero);
    Value lhsLt0 =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::lt, adaptor.getLhs(),
                                      zero);
    Value rhsLt0 =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::lt, adaptor.getRhs(),
                                      zero);
    Value signsDifferent =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::ne, lhsLt0, rhsLt0);
    Value adjust =
        rewriter.create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
                                             rNeZero, signsDifferent);

    Value qMinusOne = rewriter.create<emitc::SubOp>(loc, dstTy, q0, one);
    Value result = rewriter.create<emitc::ConditionalOp>(loc, dstTy, adjust,
                                                         qMinusOne, q0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithShiftLeftToEmitC : public OpConversionPattern<arith::ShLIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ShLIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    if (bitWidth == 1) {
      // Compute on u8 and truncate to i1.
      auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
      Value lhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getLhs());
      Value rhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getRhs());
      Value sh = rewriter.create<emitc::BitwiseLeftShiftOp>(loc, u8Ty, lhsU8,
                                                            rhsU8);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, u8Ty, sh,
                                               makeEmitCIntConstant(rewriter, loc,
                                                                    u8Ty, 1));
      rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value shU =
        rewriter.create<emitc::BitwiseLeftShiftOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, shU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithShiftRightUIToEmitC : public OpConversionPattern<arith::ShRUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ShRUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    if (bitWidth == 1) {
      // (x >> y) on i1 is either x (y==0) or 0 (y!=0); approximate in u8.
      auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
      Value lhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getLhs());
      Value rhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getRhs());
      Value sh = rewriter.create<emitc::BitwiseRightShiftOp>(loc, u8Ty, lhsU8,
                                                             rhsU8);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, u8Ty, sh,
                                               makeEmitCIntConstant(rewriter, loc,
                                                                    u8Ty, 1));
      rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value shU =
        rewriter.create<emitc::BitwiseRightShiftOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, shU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithShiftRightSIToEmitC : public OpConversionPattern<arith::ShRSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ShRSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    if (bitWidth == 1) {
      // (x >> y) on i1 is either x (y==0) or 0 (y!=0); approximate in u8.
      auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
      Value lhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getLhs());
      Value rhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getRhs());
      Value sh = rewriter.create<emitc::BitwiseRightShiftOp>(loc, u8Ty, lhsU8,
                                                             rhsU8);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, u8Ty, sh,
                                               makeEmitCIntConstant(rewriter, loc,
                                                                    u8Ty, 1));
      rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
      return success();
    }

    // Signed arithmetic shift; cast RHS to unsigned to interpret shift amount.
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value sh =
        rewriter.create<emitc::BitwiseRightShiftOp>(loc, dstTy, adaptor.getLhs(),
                                                    rhsU);
    rewriter.replaceOp(op, sh);
    return success();
  }
};

struct ArithNegFToEmitC : public OpConversionPattern<arith::NegFOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::NegFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::UnaryMinusOp>(op, dstTy, adaptor.getOperand());
    return success();
  }
};

struct ArithRemFToEmitC : public OpConversionPattern<arith::RemFOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::RemFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // Use builtin `fmod` when possible. For f16, compute in float and cast back.
    Type callTy = dstTy;
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (auto opFloatTy = dyn_cast<FloatType>(op.getType())) {
      if (opFloatTy.isF16()) {
        auto f32Ty = emitc::OpaqueType::get(rewriter.getContext(), "float");
        lhs = emitCCast(rewriter, loc, f32Ty, lhs);
        rhs = emitCCast(rewriter, loc, f32Ty, rhs);
        callTy = f32Ty;
      }
    }

    // Prefer `__builtin_fmod*` to avoid relying on extra headers.
    llvm::StringRef callee = "__builtin_fmod";
    if (auto opFloatTy = dyn_cast<FloatType>(op.getType())) {
      if (opFloatTy.isF32() || opFloatTy.isF16())
        callee = "__builtin_fmodf";
      else if (opFloatTy.isF64())
        callee = "__builtin_fmod";
    }

    auto call = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{callTy}, callee, ValueRange{lhs, rhs},
        /*args=*/ArrayAttr{}, /*template_args=*/ArrayAttr{});
    Value result = call.getResult(0);
    if (callTy != dstTy)
      result = emitCCast(rewriter, loc, dstTy, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithSelectToEmitC : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getCondition().getType().isInteger(1))
      return rewriter.notifyMatchFailure(
          op, "only scalar i1 conditions supported for arith.select");

    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    auto cond =
        rewriter.create<emitc::ConditionalOp>(op.getLoc(), dstTy,
                                              adaptor.getCondition(),
                                              adaptor.getTrueValue(),
                                              adaptor.getFalseValue());
    rewriter.replaceOp(op, cond.getResult());
    return success();
  }
};

struct ArithExtUIToEmitC : public OpConversionPattern<arith::ExtUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    // i1 -> iN: bool to integer already behaves as 0/1.
    if (srcIntTy.getWidth() == 1) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    auto uDstTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), dstIntTy.getWidth());
    Value srcU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getIn(),
                                           srcIntTy.getWidth());
    Value extU = emitCCast(rewriter, loc, uDstTy, srcU);
    Value result = emitCCast(rewriter, loc, dstTy, extU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithExtSIToEmitC : public OpConversionPattern<arith::ExtSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ExtSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    // i1 sign-extension: 0 -> 0, 1 -> -1.
    if (srcIntTy.getWidth() == 1) {
      Value zero = makeEmitCIntConstant(rewriter, loc, dstTy, 0);
      Value asInt = emitCCast(rewriter, loc, dstTy, adaptor.getIn());
      Value neg = rewriter.create<emitc::SubOp>(loc, dstTy, zero, asInt).getResult();
      rewriter.replaceOp(op, neg);
      return success();
    }

    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
    return success();
  }
};

template <typename CastOp>
struct ArithCastToEmitC : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(CastOp op, typename CastOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
    return success();
  }
};

struct ArithIndexCastUIToEmitC : public OpConversionPattern<arith::IndexCastUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::IndexCastUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // MemRef casts are handled elsewhere; for safety, fall back to emitc.cast.
    if (isa<MemRefType>(op.getIn().getType()) || isa<MemRefType>(op.getType())) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    auto getBW = [](Type t) -> std::optional<unsigned> {
      if (auto i = dyn_cast<IntegerType>(t))
        return i.getWidth();
      if (isa<IndexType>(t))
        return kPTOIndexBitWidth;
      return std::nullopt;
    };

    auto srcBW = getBW(op.getIn().getType());
    auto dstBW = getBW(op.getType());
    if (!srcBW || !dstBW)
      return rewriter.notifyMatchFailure(op, "unsupported index_castui types");

    if (*dstBW <= *srcBW) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    auto uSrcTy = getUnsignedIntOpaqueType(rewriter.getContext(), *srcBW);
    auto uDstTy = getUnsignedIntOpaqueType(rewriter.getContext(), *dstBW);
    Value srcU = emitCCast(rewriter, loc, uSrcTy, adaptor.getIn());
    Value extU = emitCCast(rewriter, loc, uDstTy, srcU);
    Value result = emitCCast(rewriter, loc, dstTy, extU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithUIToFPToEmitC : public OpConversionPattern<arith::UIToFPOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::UIToFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer input");

    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // Convert via an unsigned integer type of the same width.
    if (srcIntTy.getWidth() == 1) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }
    Value srcU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getIn(),
                                           srcIntTy.getWidth());
    Value fp = rewriter.create<emitc::CastOp>(loc, dstTy, srcU).getResult();
    rewriter.replaceOp(op, fp);
    return success();
  }
};

struct ArithFPToUIToEmitC : public OpConversionPattern<arith::FPToUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::FPToUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    if (!dstIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer result");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    auto uDstTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), dstIntTy.getWidth());
    Value asU = rewriter.create<emitc::CastOp>(loc, uDstTy, adaptor.getIn()).getResult();
    Value result = emitCCast(rewriter, loc, dstTy, asU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithBitcastToEmitC : public OpConversionPattern<arith::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::BitcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // For pointer-like types, a regular cast is fine.
    if (isa<emitc::PointerType>(dstTy)) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    // Only support scalar int/float/index bitcasts here.
    auto srcTy = op.getIn().getType();
    auto dstOrigTy = op.getType();

    auto getBitWidth = [](Type t) -> std::optional<unsigned> {
      if (auto it = dyn_cast<IntegerType>(t))
        return it.getWidth();
      if (auto ft = dyn_cast<FloatType>(t))
        return ft.getWidth();
      if (isa<IndexType>(t))
        return kPTOIndexBitWidth;
      return std::nullopt;
    };
    auto srcBW = getBitWidth(srcTy);
    auto dstBW = getBitWidth(dstOrigTy);
    if (!srcBW || !dstBW || *srcBW != *dstBW)
      return rewriter.notifyMatchFailure(op, "bitcast requires equal bitwidth");

    // Determine the template argument from the destination type string.
    auto dstOpaque = dyn_cast<emitc::OpaqueType>(dstTy);
    if (!dstOpaque)
      return rewriter.notifyMatchFailure(op, "expected emitc opaque dest type");

    auto templateArgs =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                      dstOpaque.getValue())});
    auto call = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{dstTy}, "ptoas_bitcast", /*operands=*/ValueRange{adaptor.getIn()},
        /*args=*/ArrayAttr{}, /*template_args=*/templateArgs);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

// arith.cmpf lowering with ordered/unordered semantics.
struct ArithCmpFToEmitC : public OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;

  struct CmpFConfig {
    bool unordered = false;
    emitc::CmpPredicate predicate = emitc::CmpPredicate::eq;
  };

  static Value isNaN(ConversionPatternRewriter &rewriter, Location loc,
                     Value v) {
    return rewriter
        .create<emitc::CmpOp>(loc, rewriter.getI1Type(), emitc::CmpPredicate::ne,
                              v, v)
        .getResult();
  }

  static Value isNotNaN(ConversionPatternRewriter &rewriter, Location loc,
                        Value v) {
    return rewriter
        .create<emitc::CmpOp>(loc, rewriter.getI1Type(), emitc::CmpPredicate::eq,
                              v, v)
        .getResult();
  }

  static std::optional<Value> buildSpecialCmpFResult(
      arith::CmpFPredicate predicate, ConversionPatternRewriter &rewriter,
      Location loc, Type i1Ty, Value lhs, Value rhs) {
    switch (predicate) {
    case arith::CmpFPredicate::AlwaysFalse:
      return makeEmitCOpaqueConstant(rewriter, loc, i1Ty, "false");
    case arith::CmpFPredicate::AlwaysTrue:
      return makeEmitCOpaqueConstant(rewriter, loc, i1Ty, "true");
    case arith::CmpFPredicate::ORD:
      return rewriter.create<emitc::LogicalAndOp>(
                 loc, i1Ty, isNotNaN(rewriter, loc, lhs),
                 isNotNaN(rewriter, loc, rhs))
          .getResult();
    case arith::CmpFPredicate::UNO:
      return rewriter.create<emitc::LogicalOrOp>(
                 loc, i1Ty, isNaN(rewriter, loc, lhs),
                 isNaN(rewriter, loc, rhs))
          .getResult();
    default:
      return std::nullopt;
    }
  }

  static std::optional<CmpFConfig>
  getCmpFConfig(arith::CmpFPredicate predicate) {
    switch (predicate) {
    case arith::CmpFPredicate::OEQ:
      return CmpFConfig{false, emitc::CmpPredicate::eq};
    case arith::CmpFPredicate::OGT:
      return CmpFConfig{false, emitc::CmpPredicate::gt};
    case arith::CmpFPredicate::OGE:
      return CmpFConfig{false, emitc::CmpPredicate::ge};
    case arith::CmpFPredicate::OLT:
      return CmpFConfig{false, emitc::CmpPredicate::lt};
    case arith::CmpFPredicate::OLE:
      return CmpFConfig{false, emitc::CmpPredicate::le};
    case arith::CmpFPredicate::ONE:
      return CmpFConfig{false, emitc::CmpPredicate::ne};
    case arith::CmpFPredicate::UEQ:
      return CmpFConfig{true, emitc::CmpPredicate::eq};
    case arith::CmpFPredicate::UGT:
      return CmpFConfig{true, emitc::CmpPredicate::gt};
    case arith::CmpFPredicate::UGE:
      return CmpFConfig{true, emitc::CmpPredicate::ge};
    case arith::CmpFPredicate::ULT:
      return CmpFConfig{true, emitc::CmpPredicate::lt};
    case arith::CmpFPredicate::ULE:
      return CmpFConfig{true, emitc::CmpPredicate::le};
    case arith::CmpFPredicate::UNE:
      return CmpFConfig{true, emitc::CmpPredicate::ne};
    default:
      return std::nullopt;
    }
  }

  static Value buildCmpFResult(const CmpFConfig &config,
                               ConversionPatternRewriter &rewriter,
                               Location loc, Type i1Ty, Value lhs, Value rhs) {
    Value cmp = rewriter
                    .create<emitc::CmpOp>(loc, i1Ty, config.predicate, lhs, rhs)
                    .getResult();
    Value unord = rewriter.create<emitc::LogicalOrOp>(
        loc, i1Ty, isNaN(rewriter, loc, lhs), isNaN(rewriter, loc, rhs));
    if (config.unordered)
      return rewriter
          .create<emitc::LogicalOrOp>(loc, i1Ty, unord, cmp)
          .getResult();
    Value ord = rewriter.create<emitc::LogicalAndOp>(
        loc, i1Ty, isNotNaN(rewriter, loc, lhs), isNotNaN(rewriter, loc, rhs));
    return rewriter
        .create<emitc::LogicalAndOp>(loc, i1Ty, ord, cmp)
        .getResult();
  }

  LogicalResult matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isa<FloatType>(op.getLhs().getType()))
      return rewriter.notifyMatchFailure(op, "cmpf only supported on scalar floats");

    auto loc = op.getLoc();
    auto i1Ty = rewriter.getI1Type();
    if (auto special = buildSpecialCmpFResult(op.getPredicate(), rewriter, loc,
                                              i1Ty, adaptor.getLhs(),
                                              adaptor.getRhs())) {
      rewriter.replaceOp(op, *special);
      return success();
    }

    auto config = getCmpFConfig(op.getPredicate());
    if (!config)
      return rewriter.notifyMatchFailure(op, "unsupported cmpf predicate");
    rewriter.replaceOp(op, buildCmpFResult(*config, rewriter, loc, i1Ty,
                                           adaptor.getLhs(), adaptor.getRhs()));
    return success();
  }
};

struct ArithAddUIExtendedToEmitC
    : public OpConversionPattern<arith::AddUIExtendedOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddUIExtendedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getSum().getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op,
                                         "expected scalar integer or index operands");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResultTypes)))
      return failure();
    if (newResultTypes.size() != 2)
      return failure();

    Type sumDstTy = newResultTypes[0];
    Type overflowDstTy = newResultTypes[1];

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    auto wideTy = getWiderUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);

    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value lhsWide = emitCCast(rewriter, loc, wideTy, lhsU);
    Value rhsWide = emitCCast(rewriter, loc, wideTy, rhsU);
    Value sumWide =
        rewriter.create<emitc::AddOp>(loc, wideTy, lhsWide, rhsWide).getResult();

    Value sumN = emitCCast(rewriter, loc, uTy, sumWide);
    Value sum = emitCCast(rewriter, loc, sumDstTy, sumN);

    Value shiftAmt = makeEmitCIntConstant(rewriter, loc, wideTy, bitWidth);
    Value high = rewriter
                     .create<emitc::BitwiseRightShiftOp>(loc, wideTy, sumWide,
                                                         shiftAmt)
                     .getResult();
    Value zeroWide = makeEmitCIntConstant(rewriter, loc, wideTy, 0);
    Value overflow =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::ne, high, zeroWide)
            .getResult();
    overflow = emitCCast(rewriter, loc, overflowDstTy, overflow);

    rewriter.replaceOp(op, {sum, overflow});
    return success();
  }
};

template <typename ArithOp, bool isUnsigned>
struct ArithMulExtendedToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getResult(0).getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op,
                                         "expected scalar integer or index operands");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    SmallVector<Type> newResultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      newResultTypes)))
      return failure();
    if (newResultTypes.size() != 2)
      return failure();

    Type lowDstTy = newResultTypes[0];
    Type highDstTy = newResultTypes[1];

    Type wideTy = isUnsigned ? (Type)getWiderUnsignedIntOpaqueType(rewriter.getContext(),
                                                                   bitWidth)
                             : (Type)getWiderSignedIntOpaqueType(rewriter.getContext(),
                                                                 bitWidth);

    Value lhsWide;
    Value rhsWide;
    if constexpr (isUnsigned) {
      Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                      bitWidth);
      Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                      bitWidth);
      lhsWide = emitCCast(rewriter, loc, wideTy, lhsU);
      rhsWide = emitCCast(rewriter, loc, wideTy, rhsU);
    } else {
      lhsWide = emitCCast(rewriter, loc, wideTy, adaptor.getLhs());
      rhsWide = emitCCast(rewriter, loc, wideTy, adaptor.getRhs());
    }

    Value prodWide =
        rewriter.create<emitc::MulOp>(loc, wideTy, lhsWide, rhsWide).getResult();
    Value low = emitCCast(rewriter, loc, lowDstTy, prodWide);

    Value shiftAmt = makeEmitCIntConstant(rewriter, loc, wideTy, bitWidth);
    Value highWide = rewriter
                         .create<emitc::BitwiseRightShiftOp>(loc, wideTy, prodWide,
                                                             shiftAmt)
                         .getResult();
    Value high = emitCCast(rewriter, loc, highDstTy, highWide);

    rewriter.replaceOp(op, {low, high});
    return success();
  }
};

using ArithMulSIExtendedToEmitC =
    ArithMulExtendedToEmitC<arith::MulSIExtendedOp, /*isUnsigned=*/false>;
using ArithMulUIExtendedToEmitC =
    ArithMulExtendedToEmitC<arith::MulUIExtendedOp, /*isUnsigned=*/true>;

struct ArithMinMaxIToEmitCBase {
  static Value makeSelect(ConversionPatternRewriter &rewriter, Location loc,
                          Type dstTy, Value cond, Value trueV, Value falseV) {
    return rewriter
        .create<emitc::ConditionalOp>(loc, dstTy, cond, trueV, falseV)
        .getResult();
  }
};

struct ArithMaxSIToEmitC : public OpConversionPattern<arith::MaxSIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MaxSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    Value cond = rewriter
                     .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                           emitc::CmpPredicate::lt,
                                           adaptor.getLhs(), adaptor.getRhs())
                     .getResult();
    Value res = makeSelect(rewriter, loc, dstTy, cond, adaptor.getRhs(),
                           adaptor.getLhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithMinSIToEmitC : public OpConversionPattern<arith::MinSIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MinSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    Value cond = rewriter
                     .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                           emitc::CmpPredicate::lt,
                                           adaptor.getLhs(), adaptor.getRhs())
                     .getResult();
    Value res = makeSelect(rewriter, loc, dstTy, cond, adaptor.getLhs(),
                           adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithMaxUIToEmitC : public OpConversionPattern<arith::MaxUIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MaxUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    Value lhsU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                           bitWidth);
    Value rhsU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                           bitWidth);
    Value cond = rewriter
                     .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                           emitc::CmpPredicate::lt, lhsU, rhsU)
                     .getResult();
    Value res = makeSelect(rewriter, loc, dstTy, cond, adaptor.getRhs(),
                           adaptor.getLhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithMinUIToEmitC : public OpConversionPattern<arith::MinUIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MinUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    Value lhsU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                           bitWidth);
    Value rhsU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                           bitWidth);
    Value cond = rewriter
                     .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                           emitc::CmpPredicate::lt, lhsU, rhsU)
                     .getResult();
    Value res = makeSelect(rewriter, loc, dstTy, cond, adaptor.getLhs(),
                           adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

// Floating-point max/min variants.
struct ArithFloatMinMaxToEmitCBase {
  static Value isNaN(ConversionPatternRewriter &rewriter, Location loc,
                     Value v) {
    return rewriter
        .create<emitc::CmpOp>(loc, rewriter.getI1Type(), emitc::CmpPredicate::ne,
                              v, v)
        .getResult();
  }

  static Value makeFZero(ConversionPatternRewriter &rewriter, Location loc,
                         Type ty) {
    return makeEmitCOpaqueConstant(rewriter, loc, ty, "0.0f");
  }
};

struct ArithMaxNumFToEmitC : public OpConversionPattern<arith::MaxNumFOp>,
                             ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MaxNumFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    Value lhsNaN = isNaN(rewriter, loc, adaptor.getLhs());
    Value rhsNaN = isNaN(rewriter, loc, adaptor.getRhs());

    Value cmpLt = rewriter
                      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                            emitc::CmpPredicate::lt,
                                            adaptor.getLhs(), adaptor.getRhs())
                      .getResult();
    Value maxNoNaN =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, cmpLt, adaptor.getRhs(),
                                          adaptor.getLhs())
            .getResult();

    Value rhsOrMax =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, rhsNaN, adaptor.getLhs(),
                                          maxNoNaN)
            .getResult();
    Value res =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, lhsNaN, adaptor.getRhs(),
                                          rhsOrMax)
            .getResult();
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithMinNumFToEmitC : public OpConversionPattern<arith::MinNumFOp>,
                             ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MinNumFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    Value lhsNaN = isNaN(rewriter, loc, adaptor.getLhs());
    Value rhsNaN = isNaN(rewriter, loc, adaptor.getRhs());

    Value cmpLt = rewriter
                      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                            emitc::CmpPredicate::lt,
                                            adaptor.getLhs(), adaptor.getRhs())
                      .getResult();
    Value minNoNaN =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, cmpLt, adaptor.getLhs(),
                                          adaptor.getRhs())
            .getResult();

    Value rhsOrMin =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, rhsNaN, adaptor.getLhs(),
                                          minNoNaN)
            .getResult();
    Value res =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, lhsNaN, adaptor.getRhs(),
                                          rhsOrMin)
            .getResult();
    rewriter.replaceOp(op, res);
    return success();
  }
};

template <typename ArithOp, bool isMaximum>
struct ArithMinMaxFPropagateNaNToEmitC : public OpConversionPattern<ArithOp>,
                                        ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  static Value buildPrimaryCandidate(ConversionPatternRewriter &rewriter,
                                     Location loc, Type dstTy, Value lhs,
                                     Value rhs) {
    Value cmpLt =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::lt, lhs, rhs)
            .getResult();
    return rewriter
        .create<emitc::ConditionalOp>(
            loc, dstTy, cmpLt, isMaximum ? rhs : lhs, isMaximum ? lhs : rhs)
        .getResult();
  }

  static Value buildSignBitValue(ConversionPatternRewriter &rewriter,
                                 Location loc, Value lhs, FloatType floatTy) {
    auto bitsTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), floatTy.getWidth());
    auto templateArgs = rewriter.getArrayAttr({emitc::OpaqueAttr::get(
        rewriter.getContext(), cast<emitc::OpaqueType>(bitsTy).getValue())});
    Value lhsBits =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, TypeRange{bitsTy}, "ptoas_bitcast",
                                         ValueRange{lhs}, ArrayAttr{},
                                         templateArgs)
            .getResult(0);
    Value oneBits = makeEmitCIntConstant(rewriter, loc, bitsTy, 1);
    Value shiftAmount =
        makeEmitCIntConstant(rewriter, loc, bitsTy, floatTy.getWidth() - 1);
    Value signMask = rewriter
                         .create<emitc::BitwiseLeftShiftOp>(loc, bitsTy, oneBits,
                                                            shiftAmount)
                         .getResult();
    return rewriter
        .create<emitc::BitwiseAndOp>(loc, bitsTy, lhsBits, signMask)
        .getResult();
  }

  static Value buildSignedZeroCandidate(ConversionPatternRewriter &rewriter,
                                        Location loc, Type dstTy, Value lhs,
                                        Value rhs, FloatType floatTy) {
    Value zero = makeFZero(rewriter, loc, dstTy);
    Value equal = rewriter
                      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                            emitc::CmpPredicate::eq, lhs, rhs)
                      .getResult();
    Value lhsZero = rewriter
                        .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                              emitc::CmpPredicate::eq, lhs,
                                              zero)
                        .getResult();
    Value bothZero = rewriter
                         .create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
                                                      equal, lhsZero)
                         .getResult();
    auto bitsTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), floatTy.getWidth());
    Value zeroBits = makeEmitCIntConstant(rewriter, loc, bitsTy, 0);
    Value lhsIsNegZero =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::ne,
                                  buildSignBitValue(rewriter, loc, lhs, floatTy),
                                  zeroBits)
            .getResult();
    Value tie = rewriter
                    .create<emitc::ConditionalOp>(
                        loc, dstTy, lhsIsNegZero, isMaximum ? rhs : lhs,
                        isMaximum ? lhs : rhs)
                    .getResult();
    return rewriter
        .create<emitc::ConditionalOp>(loc, dstTy, bothZero, tie,
                                      buildPrimaryCandidate(rewriter, loc, dstTy,
                                                            lhs, rhs))
        .getResult();
  }

  static Value buildNaNPropagatingResult(ConversionPatternRewriter &rewriter,
                                         Location loc, Type dstTy, Value lhs,
                                         Value rhs, FloatType floatTy) {
    Value lhsNaN = isNaN(rewriter, loc, lhs);
    Value rhsNaN = isNaN(rewriter, loc, rhs);
    Value noNaN =
        buildSignedZeroCandidate(rewriter, loc, dstTy, lhs, rhs, floatTy);
    Value rhsOrNoNaN = rewriter
                           .create<emitc::ConditionalOp>(loc, dstTy, rhsNaN, rhs,
                                                         noNaN)
                           .getResult();
    return rewriter
        .create<emitc::ConditionalOp>(loc, dstTy, lhsNaN, lhs, rhsOrNoNaN)
        .getResult();
  }

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<FloatType>(op.getType()))
      return rewriter.notifyMatchFailure(op, "expected scalar float type");

    auto loc = op.getLoc();
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    auto floatTy = cast<FloatType>(op.getType());
    rewriter.replaceOp(op, buildNaNPropagatingResult(
                               rewriter, loc, dstTy, adaptor.getLhs(),
                               adaptor.getRhs(), floatTy));
    return success();
  }
};

using ArithMaximumFToEmitC =
    ArithMinMaxFPropagateNaNToEmitC<arith::MaximumFOp, /*isMaximum=*/true>;
using ArithMinimumFToEmitC =
    ArithMinMaxFPropagateNaNToEmitC<arith::MinimumFOp, /*isMaximum=*/false>;

//===----------------------------------------------------------------------===//
// Arith -> EmitC helpers
//===----------------------------------------------------------------------===//

static emitc::OpaqueType getSignedIntOpaqueType(MLIRContext *ctx,
                                                unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
    return emitc::OpaqueType::get(ctx, "int8_t");
  case 8:
    return emitc::OpaqueType::get(ctx, "int8_t");
  case 16:
    return emitc::OpaqueType::get(ctx, "int16_t");
  case 32:
    return emitc::OpaqueType::get(ctx, "int32_t");
  case 64:
    return emitc::OpaqueType::get(ctx, "int64_t");
  case 128:
    return emitc::OpaqueType::get(ctx, "__int128");
  default:
    llvm::errs() << "[Debug] Unsupported signed integer bitwidth: " << bitWidth
                 << "\n";
    return emitc::OpaqueType::get(ctx, "int64_t");
  }
}

static emitc::OpaqueType getUnsignedIntOpaqueType(MLIRContext *ctx,
                                                  unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
    return emitc::OpaqueType::get(ctx, "uint8_t");
  case 8:
    return emitc::OpaqueType::get(ctx, "uint8_t");
  case 16:
    return emitc::OpaqueType::get(ctx, "uint16_t");
  case 32:
    return emitc::OpaqueType::get(ctx, "uint32_t");
  case 64:
    return emitc::OpaqueType::get(ctx, "uint64_t");
  case 128:
    return emitc::OpaqueType::get(ctx, "unsigned __int128");
  default:
    llvm::errs() << "[Debug] Unsupported unsigned integer bitwidth: "
                 << bitWidth << "\n";
    return emitc::OpaqueType::get(ctx, "uint64_t");
  }
}

static emitc::OpaqueType getWiderSignedIntOpaqueType(MLIRContext *ctx,
                                                     unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
  case 8:
    return getSignedIntOpaqueType(ctx, 16);
  case 16:
    return getSignedIntOpaqueType(ctx, 32);
  case 32:
    return getSignedIntOpaqueType(ctx, 64);
  case 64:
    return getSignedIntOpaqueType(ctx, 128);
  default:
    return getSignedIntOpaqueType(ctx, 128);
  }
}

static emitc::OpaqueType getWiderUnsignedIntOpaqueType(MLIRContext *ctx,
                                                       unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
  case 8:
    return getUnsignedIntOpaqueType(ctx, 16);
  case 16:
    return getUnsignedIntOpaqueType(ctx, 32);
  case 32:
    return getUnsignedIntOpaqueType(ctx, 64);
  case 64:
    return getUnsignedIntOpaqueType(ctx, 128);
  default:
    return getUnsignedIntOpaqueType(ctx, 128);
  }
}

static Value makeEmitCOpaqueConstant(ConversionPatternRewriter &rewriter,
                                     Location loc, Type type,
                                     llvm::StringRef literal) {
  auto attr = emitc::OpaqueAttr::get(rewriter.getContext(), literal);
  return rewriter.create<emitc::ConstantOp>(loc, type, attr);
}

static Value makeEmitCIntConstant(ConversionPatternRewriter &rewriter,
                                  Location loc, Type type, int64_t value) {
  return makeEmitCOpaqueConstant(rewriter, loc, type, std::to_string(value));
}

static FailureOr<std::string> buildEmitCOpaqueConstantLiteral(Type targetType,
                                                              Attribute valueAttr) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(targetType);
  if (!opaqueTy)
    return failure();

  if (opaqueTy.getValue() == "pto::MrgSortExecutedNumList") {
    auto dense = dyn_cast_or_null<DenseIntElementsAttr>(valueAttr);
    if (!dense)
      return failure();

    auto vecTy = dyn_cast<VectorType>(dense.getType());
    if (!vecTy || vecTy.getRank() != 1 || vecTy.getNumElements() != 4 ||
        !vecTy.getElementType().isInteger(16))
      return failure();

    std::string literal;
    llvm::raw_string_ostream os(literal);
    os << "pto::MrgSortExecutedNumList{";
    bool first = true;
    for (APInt elem : dense.getValues<APInt>()) {
      if (!first)
        os << ", ";
      first = false;
      os << elem.getZExtValue();
    }
    os << "}";
    os.flush();
    return literal;
  }

  return failure();
}

static Value emitCCast(ConversionPatternRewriter &rewriter, Location loc,
                       Type dstType, Value src) {
  if (src.getType() == dstType)
    return src;
  return rewriter.createOrFold<emitc::CastOp>(loc, dstType, src);
}

// For signless iN integers lowered to signed C++ types, this creates a value
// representing the same N-bit pattern in an unsigned C++ type of the same
// width. This avoids incorrect sign-extension when later widening to a larger
// unsigned type.
static Value castSignlessIntToUnsignedSameWidth(ConversionPatternRewriter &rewriter,
                                                Location loc, Value v,
                                                unsigned bitWidth) {
  auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
  return emitCCast(rewriter, loc, uTy, v);
}

struct ArithMulIToEmitC : public OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern<arith::MulIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    // i1 mul is equivalent to bitwise AND (mod 2 arithmetic).
    if (bitWidth == 1) {
      rewriter.replaceOpWithNewOp<emitc::BitwiseAndOp>(op, opTy, adaptor.getLhs(),
                                                      adaptor.getRhs());
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value mulU = rewriter.create<emitc::MulOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, mulU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithAddIToEmitC : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    // i1 add is equivalent to XOR (mod 2 arithmetic).
    if (bitWidth == 1) {
      rewriter.replaceOpWithNewOp<emitc::BitwiseXorOp>(op, opTy, adaptor.getLhs(),
                                                      adaptor.getRhs());
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value addU = rewriter.create<emitc::AddOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, addU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithCastOPToEmitC : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern<arith::IndexCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    if (adaptor.getIn().getType() == newTy) {
      rewriter.replaceOp(op, adaptor.getIn());
      return success();
    }
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, newTy, adaptor.getIn());
    return success();
  }
};

struct ArithSubIToEmitC : public OpConversionPattern<arith::SubIOp> {
  using OpConversionPattern<arith::SubIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::SubIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    // i1 sub is equivalent to XOR (mod 2 arithmetic).
    if (bitWidth == 1) {
      rewriter.replaceOpWithNewOp<emitc::BitwiseXorOp>(op, opTy, adaptor.getLhs(),
                                                      adaptor.getRhs());
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value subU = rewriter.create<emitc::SubOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, subU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithDivSIToEmitC : public OpConversionPattern<arith::DivSIOp> {
  using OpConversionPattern<arith::DivSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::DivSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::DivOp>(op, newTy, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

struct ArithRemSIToEmitC : public OpConversionPattern<arith::RemSIOp> {
  using OpConversionPattern<arith::RemSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::RemSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::RemOp>(op, newTy, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

struct ArithTruncIToEmitC : public OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern<arith::TruncIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    // to-i1 conversions: Arith wants truncation to the low bit, while C/C++
    // casts to bool are equivalent to `v != 0`. Implement as `(bool)(v & 1)`.
    if (dstIntTy.getWidth() == 1) {
      if (srcIntTy.getWidth() == 1) {
        rewriter.replaceOp(op, adaptor.getIn());
        return success();
      }

      auto uSrcTy =
          getUnsignedIntOpaqueType(rewriter.getContext(), srcIntTy.getWidth());
      Value inU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getIn(),
                                                     srcIntTy.getWidth());
      Value one = makeEmitCIntConstant(rewriter, loc, uSrcTy, 1);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, uSrcTy, inU, one);
      Value asBool = emitCCast(rewriter, loc, dstTy, masked);
      rewriter.replaceOp(op, asBool);
      return success();
    }

    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
    return success();
  }
};

struct ArithConstantToEmitC : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newType = getTypeConverter()->convertType(op.getType());
    if (!newType)
      return failure();

    // `adaptor.getValue()` may be null if attribute conversion isn't defined.
    // Use the original attribute as fallback and always cast null-safely.
    Attribute valueAttr = adaptor.getValue();
    if (!valueAttr)
      valueAttr = op.getValue();

    if (auto opaqueLiteral = buildEmitCOpaqueConstantLiteral(newType, valueAttr);
        succeeded(opaqueLiteral)) {
      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), *opaqueLiteral);
      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
      return success();
    }

    if (auto floatAttr = dyn_cast_or_null<FloatAttr>(valueAttr)) {
      SmallString<32> valStr;
      floatAttr.getValue().toString(valStr);
      llvm::StringRef s(valStr);
      // Ensure the literal parses as a floating-point constant in C/C++.
      // `APFloat::toString` may emit "1" for integral values; make it "1.0".
      const bool hasFloatMarker =
          s.contains('.') || s.contains('e') || s.contains('E') ||
          s.contains('p') || s.contains('P') || s.starts_with("0x") ||
          s.starts_with("0X") || s.starts_with("nan") ||
          s.starts_with("-nan") || s.starts_with("inf") ||
          s.starts_with("-inf");
      if (!hasFloatMarker)
        valStr.append(".0");
      // Suffix: keep `f` for f16/f32; omit for f64.
      if (!floatAttr.getType().isF64())
        valStr.append("f");
      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
      return success();
    }

    if (auto intAttr = dyn_cast_or_null<IntegerAttr>(valueAttr)) {
      std::string valStr = std::to_string(intAttr.getValue().getSExtValue());
      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
      return success();
    }

    return failure();
  }
};
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

    auto gatherOobTok = [&](pto::GatherOOB mode) -> StringRef {
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

    SmallVector<Attribute, 2> templateArgVec;
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

struct AffineApplyMulConstToEmitC
    : public OpConversionPattern<affine::AffineApplyOp> {
  using OpConversionPattern<affine::AffineApplyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto map = op.getAffineMap();

    if (map.getNumDims() != 0 || map.getNumSymbols() != 1)
      return failure();

    auto expr = map.getResult(0);
    auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!bin || bin.getKind() != AffineExprKind::Mul)
      return failure();

    auto lhs = bin.getLHS();
    auto rhs = bin.getRHS();

    auto symExpr = dyn_cast<AffineSymbolExpr>(lhs);
    auto constExpr = dyn_cast<AffineConstantExpr>(rhs);
    if (!symExpr || !constExpr)
      return failure();

    Value inputVal = adaptor.getMapOperands()[0];

    std::string valStr = std::to_string(constExpr.getValue());
    auto cstAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
    auto cstOp = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), inputVal.getType(), cstAttr);

    rewriter.replaceOpWithNewOp<emitc::MulOp>(
        op, inputVal.getType(), inputVal, cstOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Kernel inference helpers
//===----------------------------------------------------------------------===//

enum class KernelKind { VecAdd, Matmul, Unknown };

[[maybe_unused]] static KernelKind inferKernelKind(func::FuncOp f) {
  bool hasAdd = false;
  bool hasMM  = false;
  f.walk([&](Operation *op) {
    if (isa<mlir::pto::TAddOp>(op)) hasAdd = true;
    if (isa<mlir::pto::TMatmulOp>(op)) hasMM = true;
    if (isa<mlir::pto::TMatmulAccOp>(op)) hasMM = true;
  });
  if (hasMM)  return KernelKind::Matmul;
  if (hasAdd) return KernelKind::VecAdd;
  return KernelKind::Unknown;
}

[[maybe_unused]] static void inferTileMNK(func::FuncOp f, int &M, int &N, int &K) {
  M = 32; N = 32; K = 32;
  SmallVector<memref::SubViewOp, 4> subs;
  f.walk([&](memref::SubViewOp sv) { subs.push_back(sv); });

  auto readShape2D = [&](memref::SubViewOp sv, int &d0, int &d1) {
    auto resTy = mlir::cast<MemRefType>(sv.getResult().getType());
    if (resTy.getRank() == 2 && resTy.hasStaticShape()) {
      d0 = (int)resTy.getDimSize(0);
      d1 = (int)resTy.getDimSize(1);
    }
  };

  if (subs.empty()) return;

  int a0=32, a1=32;
  readShape2D(subs[0], a0, a1);
  M = a0; N = a1;

  if (subs.size() >= 2) {
    int b0=32, b1=32;
    readShape2D(subs[0], a0, a1);
    readShape2D(subs[1], b0, b1);
    M = a0; K = a1; N = b1;
  }
}

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

  LogicalResult matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Convert the function signature with the type converter.
    Type convertedTy = getTypeConverter()->convertType(op.getFunctionType());
    auto funcType = dyn_cast_or_null<FunctionType>(convertedTy);
    if (!funcType)
      return rewriter.notifyMatchFailure(op, "failed to convert function type");
    if (funcType.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "EmitC cannot return multiple values");

    // Create the EmitC function with the converted signature.
    auto emitcFunc =
        rewriter.create<emitc::FuncOp>(op.getLoc(), op.getName(), funcType);

    for (const auto &namedAttr : op->getAttrs()) {
      StringRef name = namedAttr.getName().strref();
      if (name == op.getFunctionTypeAttrName() ||
          name == SymbolTable::getSymbolAttrName() ||
          name == pto::kPTOEntryAttrName ||
          name == pto::kLegacyHACCEntryAttrName ||
          name == "pto.internal.entry")
        continue;
      emitcFunc->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    if (op.isDeclaration()) {
      emitcFunc.setSpecifiersAttr(
          rewriter.getStrArrayAttr({"extern \"C\"", "AICORE"}));
      rewriter.eraseOp(op);
      return success();
    }

    if (pto::isPTOEntryFunction(op)) {
      emitcFunc.setSpecifiersAttr(
          rewriter.getStrArrayAttr({"extern \"C\"", "__global__ AICORE"}));
    } else if (op.isPrivate()) {
      emitcFunc.setSpecifiersAttr(
          rewriter.getStrArrayAttr({"static", "AICORE"}));
    } else {
      emitcFunc.setSpecifiersAttr(rewriter.getStrArrayAttr({"AICORE"}));
    }

    std::optional<StringRef> kernelKindMacro = getKernelKindMacro(op);
    bool needsNoSplitGuard = needsA5NoSplitVectorGuard(op.getOperation());

    // Inline the original body, then convert region/block argument types to
    // match the converted signature (also covers CFG blocks introduced by
    // pre-lowering, e.g. scf.while -> cf.br/cf.cond_br).
    rewriter.inlineRegionBefore(op.getBody(), emitcFunc.getBody(),
                                emitcFunc.end());

    TypeConverter::SignatureConversion entryConv(op.getNumArguments());
    for (unsigned i = 0; i < op.getNumArguments(); ++i)
      entryConv.addInputs(i, funcType.getInput(i));

    if (failed(rewriter.convertRegionTypes(&emitcFunc.getBody(),
                                           *getTypeConverter(), &entryConv)))
      return failure();

    // Preserve the existing function prologue shape. `kernel_kind` functions are
    // emitted with the same macro guard/reset sequence that used to come from
    // early pto.section wrapping, but only after SCF pre-lowering has finished.
    {
      Block &entryBlock = emitcFunc.getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      rewriter.create<emitc::VerbatimOp>(op.getLoc(), "using T = float;");
      if (kernelKindMacro) {
        std::string startMacro = "\n#if defined(" + kernelKindMacro->str() + ")";
        rewriter.create<emitc::VerbatimOp>(op.getLoc(), startMacro);
        if (*kernelKindMacro == "__DAV_VEC__") {
          rewriter.create<emitc::VerbatimOp>(op.getLoc(), "set_mask_norm();");
          rewriter.create<emitc::VerbatimOp>(op.getLoc(),
                                             "set_vector_mask(-1, -1);");
          if (needsNoSplitGuard)
            rewriter.create<emitc::VerbatimOp>(
                op.getLoc(), "if (get_subblockid() == 0) {");
        }
      }
    }

    if (kernelKindMacro) {
      Block &lastBlock = emitcFunc.getBody().back();
      rewriter.setInsertionPoint(lastBlock.getTerminator());
      if (*kernelKindMacro == "__DAV_VEC__" && needsNoSplitGuard)
        rewriter.create<emitc::VerbatimOp>(op.getLoc(), "}");
      std::string endMacro = "#endif // " + kernelKindMacro->str() + "\n";
      rewriter.create<emitc::VerbatimOp>(op.getLoc(), endMacro);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SubView lowering to GlobalTensor (keep your existing code)
//===----------------------------------------------------------------------===

enum class Role { A, B, C, Unknown };

template <typename MatmulLikeOp>
static std::optional<Role> inferMatmulLikeSubviewRole(MatmulLikeOp op,
                                                      Value buffer) {
  if (op.getLhs() == buffer)
    return Role::A;
  if (op.getRhs() == buffer)
    return Role::B;
  return std::nullopt;
}

static std::optional<Role> inferSubviewRoleFromLoadUser(mlir::pto::TLoadOp load) {
  Value buffer = load.getDst();
  if (!buffer)
    return std::nullopt;
  for (Operation *user : buffer.getUsers()) {
    if (auto matmul = dyn_cast<mlir::pto::TMatmulOp>(user)) {
      if (auto role = inferMatmulLikeSubviewRole(matmul, buffer))
        return role;
      continue;
    }
    if (auto matmulAcc = dyn_cast<mlir::pto::TMatmulAccOp>(user)) {
      if (auto role = inferMatmulLikeSubviewRole(matmulAcc, buffer))
        return role;
    }
  }
  return std::nullopt;
}

static std::optional<Role> inferSubviewRoleFromUser(Operation *user, Value result) {
  if (auto load = dyn_cast<mlir::pto::TLoadOp>(user))
    return inferSubviewRoleFromLoadUser(load);
  if (auto store = dyn_cast<mlir::pto::TStoreOp>(user)) {
    if (store.getDst() == result)
      return Role::C;
  }
  return std::nullopt;
}

[[maybe_unused]] static Role inferSubviewRole(memref::SubViewOp sv) {
  Value result = sv.getResult();
  for (Operation *user : result.getUsers()) {
    if (auto role = inferSubviewRoleFromUser(user, result))
      return *role;
  }
  return Role::Unknown;
}

// =============================================================================
// 4. MemRef SubView -> Explicit Shape/Stride Construction (Full Implementation)
// =============================================================================
struct SubviewToEmitCPattern : public OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern<memref::SubViewOp>::OpConversionPattern;

  // 辅助函数：尝试从 OpFoldResult 中提取静态整数值
  std::optional<int64_t> extractStaticInt(OpFoldResult ofr) const {
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr))
        return intAttr.getInt();
    } else {
      Value v = ofr.get<Value>();
      if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto iAttr = dyn_cast<IntegerAttr>(cOp.getValue()))
          return iAttr.getInt();
      } else if (auto idxOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
        return idxOp.value();
      }
    }
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    
    // 获取源 MemRef 类型信息
    auto srcType = mlir::cast<MemRefType>(op.getSource().getType());
    int64_t rank = srcType.getRank();

	    auto elemTypeToString = [&](Type elemTy) -> std::string {
	      if (elemTy.isF16())
	        return "half";
	      if (elemTy.isBF16())
	        return "bfloat16_t";
	      if (elemTy.isF32())
	        return "float";
	      if (elemTy.isF64())
	        return "double";
      if (elemTy.isInteger(8)) {
        if (elemTy.isSignlessInteger(8) || elemTy.isSignedInteger(8))
          return "int8_t";
        return "uint8_t";
      }
      if (elemTy.isInteger(16)) {
        if (elemTy.isSignlessInteger(16) || elemTy.isSignedInteger(16))
          return "int16_t";
        return "uint16_t";
      }
      if (elemTy.isInteger(32)) {
        if (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
          return "int32_t";
        return "uint32_t";
      }
      if (elemTy.isInteger(64)) {
        return cast<IntegerType>(elemTy).isUnsigned() ? "uint64_t" : "int64_t";
      }
      return "float";
    };

    // -------------------------------------------------------------------------
    // Part 1: 指针偏移计算 (Runtime Pointer Arithmetic)
    // -------------------------------------------------------------------------
    
    // 准备类型: unsigned
    Type u32Ty = emitc::OpaqueType::get(ctx, "unsigned");
    
    // Helper: 创建 unsigned 常量
    auto mkU32 = [&](int64_t v) -> Value {
      return rewriter.create<emitc::ConstantOp>(
          loc, u32Ty, emitc::OpaqueAttr::get(ctx, std::to_string(v)));
    };

    // Helper: 将 OpFoldResult 转为 EmitC Value (用于计算)
    auto ofrToEmitCValue = [&](OpFoldResult ofr) -> Value {
      if (auto v = ofr.dyn_cast<Value>()) {
        Value rv = rewriter.getRemappedValue(v);
        // 如果类型不匹配，插入 Cast
        if (rv.getType() != u32Ty)
             return rewriter.create<emitc::CastOp>(loc, u32Ty, rv).getResult();
        return rv;
      }
      if (auto attr = ofr.dyn_cast<Attribute>()) {
         if (auto ia = dyn_cast<IntegerAttr>(attr))
             return mkU32(ia.getValue().getSExtValue());
      }
      return mkU32(0);
    };

    // 1. 获取 Source 的 Strides (支持动态 Stride 收集)
    SmallVector<OpFoldResult> sourceStrides;

    if (auto rc = op.getSource().getDefiningOp<memref::ReinterpretCastOp>()) {
        sourceStrides = rc.getMixedStrides();
    } else {
        SmallVector<int64_t> strideInts;
        int64_t offset = ShapedType::kDynamic;
        bool useTypeStrides = succeeded(getStridesAndOffset(srcType, strideInts, offset));
        (void)offset;
        if (useTypeStrides) {
          for (int64_t s : strideInts) {
            if (s == ShapedType::kDynamic)
              useTypeStrides = false;
          }
        }
        if (useTypeStrides) {
            for (int64_t s : strideInts) {
                sourceStrides.push_back(rewriter.getIndexAttr(s));
            }
        } else {
            // Fallback: Compact Layout
            auto shape = srcType.getShape();
            int64_t current = 1;
            sourceStrides.resize(rank);
            for (int i = rank - 1; i >= 0; --i) {
                sourceStrides[i] = rewriter.getIndexAttr(current);
                if (shape[i] != ShapedType::kDynamic) current *= shape[i];
            }
        }
    }

    // 2. 计算运行时 Offset
    auto staticOffsets = op.getStaticOffsets();
    auto dynamicOffsets = adaptor.getOffsets();
    int dynOffIdx = 0;
    Value totalOffset = mkU32(0);

    for (int i = 0; i < rank; ++i) {
        // A. 获取 Offset
        Value offVal;
        if (staticOffsets[i] == ShapedType::kDynamic) {
            Value rawDyn = dynamicOffsets[dynOffIdx++];
            offVal = rewriter.create<emitc::CastOp>(loc, u32Ty, rawDyn);
        } else {
            offVal = mkU32(staticOffsets[i]);
        }

        // B. 获取 Stride (用于指针计算)
        Value strideVal = mkU32(1);
        if (i < (int)sourceStrides.size()) {
            strideVal = ofrToEmitCValue(sourceStrides[i]);
        }

        // C. 累加
        Value term = rewriter.create<emitc::MulOp>(loc, u32Ty, offVal, strideVal);
        totalOffset = rewriter.create<emitc::AddOp>(loc, u32Ty, totalOffset, term);
    }

    // 3. 生成新指针
    //
    // NOTE: Some toolchains may materialize kernel pointer params as `void*` even
    // when the underlying element type is i16. Pointer arithmetic on `void*`
    // is ill-formed in C++, so we explicitly cast to a typed pointer for i16.
    Value sourcePtr = adaptor.getSource();
    Value tileCandidate = sourcePtr;
    if (auto castOp = sourcePtr.getDefiningOp<emitc::CastOp>()) {
      tileCandidate = castOp.getOperand();
    } else if (auto uc =
                   sourcePtr.getDefiningOp<UnrealizedConversionCastOp>()) {
      tileCandidate = uc.getOperand(0);
    }
    if (auto ot = dyn_cast<emitc::OpaqueType>(tileCandidate.getType())) {
      auto tyStr = ot.getValue();
      if (tyStr.find("Tile<") != std::string::npos ||
          tyStr.find("ConvTile<") != std::string::npos) {
        std::string elemTok = elemTypeToString(srcType.getElementType());
        pto::AddressSpace as = pto::AddressSpace::GM;
        if (auto asAttr =
                dyn_cast_or_null<pto::AddressSpaceAttr>(srcType.getMemorySpace()))
          as = asAttr.getAddressSpace();
        sourcePtr =
            materializeTileDataValue(rewriter, loc, tileCandidate, as, elemTok);
        if (tileDataReturnsIntegralAddress(as))
          sourcePtr =
              materializeAddressAsPointer(rewriter, loc, sourcePtr, as, elemTok);
      }
    }
    Value newPtr;
    {
      auto resTy = mlir::cast<MemRefType>(op.getResult().getType());
      Type elemTy = resTy.getElementType();
      if (elemTy.isInteger(16)) {
        std::string castElemTypeStr = "int16_t";
        if (cast<IntegerType>(elemTy).isUnsigned())
          castElemTypeStr = "uint16_t";

        std::string qualifier = "__gm__";
        if (Attribute ms = srcType.getMemorySpace()) {
          if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(ms)) {
            qualifier = addrSpaceQualifier(ptoAttr.getAddressSpace());
          }
        }

        auto typedPtrTy = emitc::PointerType::get(
            emitc::OpaqueType::get(ctx, qualifier + " " + castElemTypeStr));
        Value typedSourcePtr = rewriter.create<emitc::CastOp>(loc, typedPtrTy, sourcePtr);
        newPtr = rewriter.create<emitc::AddOp>(loc, typedPtrTy, typedSourcePtr, totalOffset);
      } else {
        newPtr = rewriter.create<emitc::AddOp>(loc, sourcePtr.getType(), sourcePtr, totalOffset);
      }
    }


    // -------------------------------------------------------------------------
    // Part 2: For non-GM memrefs, keep pointer (no GlobalTensor).
    // -------------------------------------------------------------------------
    bool isGlobal = true;
    if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(srcType.getMemorySpace())) {
      auto as = asAttr.getAddressSpace();
      isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
    }
    if (!isGlobal) {
      Type dstTy = getTypeConverter()->convertType(op.getType());
      if (!dstTy)
        return failure();
      if (newPtr.getType() != dstTy)
        newPtr = rewriter.create<emitc::CastOp>(loc, dstTy, newPtr);
      rewriter.replaceOp(op, newPtr);
      return success();
    }

    // -------------------------------------------------------------------------
    // Part 3: 生成 GlobalTensor 类型 (Shape/Stride Template Generation)
    // -------------------------------------------------------------------------
    
    // When emitting C++ with `declareVariablesAtTop`, value declarations are
    // hoisted before body statements. Avoid introducing local `using` aliases
    // for templated types (Shape/Stride/GlobalTensor) because those aliases
    // would appear after the hoisted declarations and break compilation
    // (`unknown type name`).
    //
    // Instead, use the fully spelled template types as EmitC opaque types.

    auto resTy = mlir::cast<MemRefType>(op.getResult().getType());
    
    // 1. 解析具体元素类型
    std::string elemTypeStr = getElemTypeStringForGT(resTy.getElementType());

    // 2. 生成 Shape 模板参数，之后会右对齐有效维度并补齐到 5 维（高维填 1）
    SmallVector<int64_t> shapeParamsVec;
    SmallVector<Value> sizeValues; // 每个维度对应的运行时 size（统一为 unsigned）
    auto resShape = resTy.getShape();
    auto mixedSizes = op.getMixedSizes();
    sizeValues.reserve(rank);
    for (int i = 0; i < resTy.getRank(); ++i) {
      if (resShape[i] == ShapedType::kDynamic) {
        shapeParamsVec.push_back(-1);
      } else {
        shapeParamsVec.push_back(resShape[i]);
      }
      // size 值：优先从 op.getMixedSizes() 取（可动态/静态），否则退化为类型里的静态 shape。
      if (i < (int)mixedSizes.size())
        sizeValues.push_back(ofrToEmitCValue(mixedSizes[i]));
      else
        sizeValues.push_back(
            mkU32(resShape[i] == ShapedType::kDynamic ? 1 : resShape[i]));
    }

    // 3. 生成 Stride 模板参数 + 运行时 stride 值（考虑 subview step）
    SmallVector<int64_t> strideTemplateVec;
    SmallVector<Value> strideValues; // 每个维度对应的运行时 stride（统一为 unsigned）
    strideTemplateVec.reserve(rank);
    strideValues.reserve(rank);
    auto subViewSteps = op.getMixedStrides();
    for (int i = 0; i < rank; ++i) {
      OpFoldResult srcStrideOfr =
          (i < (int)sourceStrides.size()) ? sourceStrides[i]
                                          : rewriter.getIndexAttr(1);
      OpFoldResult stepOfr = (i < (int)subViewSteps.size())
                                 ? subViewSteps[i]
                                 : rewriter.getIndexAttr(1);

      auto srcStatic = extractStaticInt(srcStrideOfr);
      auto stepStatic = extractStaticInt(stepOfr);
      if (srcStatic && stepStatic) {
        int64_t finalStride = (*srcStatic) * (*stepStatic);
        strideTemplateVec.push_back(finalStride);
        strideValues.push_back(mkU32(finalStride));
        continue;
      }

      strideTemplateVec.push_back(-1);
      Value srcV = ofrToEmitCValue(srcStrideOfr);
      Value stepV = ofrToEmitCValue(stepOfr);
      // 尽量避免乘以 1 生成冗余指令
      if (stepStatic && *stepStatic == 1)
        strideValues.push_back(srcV);
      else if (srcStatic && *srcStatic == 1)
        strideValues.push_back(stepV);
      else
        strideValues.push_back(
            rewriter.create<emitc::MulOp>(loc, u32Ty, srcV, stepV));
    }

    // 3.1 右对齐到 5 维：shape 补 1；已有维度继承原 stride；
    //      被补出来的高维按“紧密升维”规则连续推导：stride[i] = shape[i+1] * stride[i+1]
    SmallVector<int64_t, 5> finalShape;
    SmallVector<int64_t, 5> finalStride;
    buildGlobalTensorShapeAndStride(shapeParamsVec, strideTemplateVec,
                                    finalShape, finalStride);
    Value oneU32 = mkU32(1);
    SmallVector<Value, 5> finalShapeValues(5, oneU32);
    SmallVector<Value, 5> finalStrideValues(5, oneU32);
    int shift = 5 - rank;

    // 先放入原始 shape/stride（保持用户提供的值）
    for (int i = 0; i < rank && i < 5; ++i) {
      finalShapeValues[shift + i] = sizeValues[i];
      finalStrideValues[shift + i] = strideValues[i];
    }

    // 从低维到高维倒推补齐 stride（仅对补出来的前置维度生效）
    for (int i = 3; i >= 0; --i) {
      // 如果该维已由原始 rank 覆盖，则保持原值
      if (i >= shift)
        continue;
      if (finalStride[i] != -1) {
        finalStrideValues[i] = mkU32(finalStride[i]);
        continue;
      }
      // 动态推导：stride[i] = shape[i+1] * stride[i+1]
      if (finalShape[i + 1] == 1) {
        finalStrideValues[i] = finalStrideValues[i + 1];
      } else {
        finalStrideValues[i] = rewriter.create<emitc::MulOp>(
            loc, u32Ty, finalShapeValues[i + 1], finalStrideValues[i + 1]);
      }
    }

    std::string shapeParams = joinIntTemplateParams(finalShape);
    std::string strideParams = joinIntTemplateParams(finalStride);

    // Spelled-out C++ types.
    std::string shapeCppType = "pto::Shape<" + shapeParams + ">";
    std::string strideCppType = "pto::Stride<" + strideParams + ">";

    // 3.0 Layout: prefer the attribute from InferPTOLayout; only fall back to
    // local inference when the pass is disabled.
    std::string layoutEnum = "pto::Layout::ND";
    if (auto layout = resolveLayoutForGlobalTensor(op, op.getSource())) {
      layoutEnum = layoutToEmitCString(*layout);
    } else {
      bool allStatic =
          llvm::all_of(finalShape, [](int64_t value) { return value != -1; }) &&
          llvm::all_of(finalStride, [](int64_t value) { return value != -1; });

      int layoutTag = 0; // ND
      auto elemBytes = 4; // default float
      if (elemTypeStr.find("half") != std::string::npos ||
          elemTypeStr.find("f16") != std::string::npos ||
          elemTypeStr.find("bf16") != std::string::npos)
        elemBytes = 2;
      else if (elemTypeStr.find("double") != std::string::npos ||
               elemTypeStr.find("f64") != std::string::npos)
        elemBytes = 8;

      if (allStatic) {
        if (finalShape[2] == 16 &&
            finalShape[2] * finalShape[3] * elemBytes == 512 &&
            finalStride[4] == 1 && finalStride[3] == finalShape[4]) {
          layoutTag = 2; // NZ
        } else {
          bool isRow = finalStride[4] == 1;
          for (int i = 3; i >= 0; --i)
            isRow &= (finalStride[i] ==
                      multiplyOrDynamic(finalStride[i + 1], finalShape[i + 1]));
          bool isCol = finalStride[0] == 1;
          for (int i = 0; i < 4; ++i)
            isCol &= (finalStride[i + 1] ==
                      multiplyOrDynamic(finalStride[i], finalShape[i]));
          if (isCol)
            layoutTag = 1; // DN
          else
            layoutTag = isRow ? 0 : 0; // fallback ND
        }
      }

      if (layoutTag == 1)
        layoutEnum = "pto::Layout::DN";
      else if (layoutTag == 2)
        layoutEnum = "pto::Layout::NZ";
    }
    // GlobalTensor takes a Layout non-type template parameter; directly use the
    // enum constant.


    // -------------------------------------------------------------------------
    // Part 3: 显式对象实例化 (Explicit Object Instantiation)
    // -------------------------------------------------------------------------

    // A. Instantiate Shape object.
    auto shapeTypeOpaque = emitc::OpaqueType::get(ctx, shapeCppType);
    SmallVector<Value> shapeArgs;
    // 从 adaptor.getSizes() 获取 subview 的所有 dynamic sizes
    for (Value dynSize : adaptor.getSizes()) {
        shapeArgs.push_back(dynSize);
    }
    
    auto shapeInstOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, 
        shapeTypeOpaque, // 返回类型
        shapeCppType,    // 调用的“函数名”即类名构造函数
        /*args=*/ArrayAttr{}, 
        /*templateArgs=*/ArrayAttr{}, 
        /*operands=*/ValueRange(shapeArgs)
    );
    
    // B. Instantiate Stride object.
    auto strideTypeOpaque = emitc::OpaqueType::get(ctx, strideCppType);
    // 仅传入动态 stride 维度对应的值，匹配 pto::Stride 的 N-parameter ctor（并满足其 static_assert）。
    SmallVector<Value> strideCtorArgs;
    strideCtorArgs.reserve(5);
    for (int i = 0; i < 5; ++i) {
      if (finalStride[i] == -1)
        strideCtorArgs.push_back(finalStrideValues[i]);
    }
    auto strideInstOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, strideTypeOpaque, strideCppType,
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(strideCtorArgs));

    // C. Instantiate GlobalTensor object (ptr + shape + stride).
    std::string gtCppType = "GlobalTensor<" + elemTypeStr + ", " + shapeCppType +
                            ", " + strideCppType + ", " + layoutEnum + ">";
    auto gtType = emitc::OpaqueType::get(ctx, gtCppType);

    // 准备构造参数: [ptr, shape_instance, stride_instance]
    SmallVector<Value> gtConstructorArgs;
    gtConstructorArgs.push_back(newPtr);
    gtConstructorArgs.push_back(shapeInstOp.getResult(0)); // 拿到 shape_inst 的 SSA Value
    gtConstructorArgs.push_back(strideInstOp.getResult(0)); // 拿到 stride_inst 的 SSA Value

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, 
        gtType, 
        gtCppType,
        /*args=*/ArrayAttr{}, 
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(gtConstructorArgs)
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper: build GlobalTensor from a static MemRef (for TLOAD/TSTORE)
//===----------------------------------------------------------------------===//

static std::string getElemTypeStringForGT(Type elemTy) {
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

static Value applyStaticMemrefOffset(ConversionPatternRewriter &rewriter,
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

static int64_t multiplyOrDynamic(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0)
    return -1;
  return lhs * rhs;
}

static void buildGlobalTensorShapeAndStride(ArrayRef<int64_t> shape,
                                            ArrayRef<int64_t> strides,
                                            SmallVectorImpl<int64_t> &shape5D,
                                            SmallVectorImpl<int64_t> &stride5D) {
  shape5D.assign(5, 1);
  stride5D.assign(5, 1);
  int rank = static_cast<int>(shape.size());
  int shift = 5 - rank;
  for (int i = 0; i < rank && i < 5; ++i) {
    shape5D[shift + i] = shape[i];
    stride5D[shift + i] = strides[i];
  }
  for (int i = 3; i >= 0; --i) {
    if (i >= shift)
      continue;
    stride5D[i] = multiplyOrDynamic(shape5D[i + 1], stride5D[i + 1]);
  }
}

static std::string joinIntTemplateParams(ArrayRef<int64_t> values) {
  std::string result;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      result += ", ";
    result += std::to_string(values[i]);
  }
  return result;
}

static SmallVector<int64_t> buildRowMajorStrides(ArrayRef<int64_t> shape) {
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

static std::string getGlobalTensorTypeStringFromShapeAndStrides(
    Type elemTy, ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
    StringRef layoutEnum) {
  SmallVector<int64_t, 5> shape5D;
  SmallVector<int64_t, 5> stride5D;
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
  if (shape5D[2] == 16 && multiplyOrDynamic(shape5D[2], shape5D[3]) * elemBytes == 512 &&
      stride5D[4] == 1 && stride5D[3] == shape5D[4]) {
    return "pto::Layout::NZ";
  }

  bool isRowMajor = stride5D[4] == 1;
  for (int i = 3; i >= 0 && isRowMajor; --i)
    isRowMajor = stride5D[i] == multiplyOrDynamic(stride5D[i + 1], shape5D[i + 1]);

  bool isColMajor = stride5D[0] == 1;
  for (int i = 0; i < 4 && isColMajor; ++i)
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
  std::string suffix = "_" + std::to_string(reinterpret_cast<uintptr_t>(anchor));
  return {
      "GTShape" + suffix,
      "GTStride" + suffix,
      "GT" + suffix,
      "GT" + suffix + "_layout",
  };
}
static Value buildGlobalTensorFromMemref(ConversionPatternRewriter &rewriter,
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
  SmallVector<int64_t, 5> shape5D;
  SmallVector<int64_t, 5> stride5D;
  buildGlobalTensorShapeAndStride(shape, strides, shape5D, stride5D);

  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + names.shapeTypeName + " = pto::Shape<" +
               joinIntTemplateParams(shape5D) + ">;");
  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + names.strideTypeName + " = pto::Stride<" +
               joinIntTemplateParams(stride5D) + ">;");

  std::string layoutEnum = resolveGlobalTensorLayout(
      anchor, basePtr, shape5D, stride5D, mrTy.getElementType());
  rewriter.create<emitc::VerbatimOp>(loc, "constexpr pto::Layout " +
                                              names.layoutConstName + " = " +
                                              layoutEnum + ";");

  auto shapeTypeOpaque = emitc::OpaqueType::get(ctx, names.shapeTypeName);
  auto strideTypeOpaque = emitc::OpaqueType::get(ctx, names.strideTypeName);
  auto shapeInstOp = rewriter.create<emitc::CallOpaqueOp>(
      loc, shapeTypeOpaque, names.shapeTypeName, ArrayAttr{}, ArrayAttr{},
      ValueRange{});
  auto strideInstOp = rewriter.create<emitc::CallOpaqueOp>(
      loc, strideTypeOpaque, names.strideTypeName, ArrayAttr{}, ArrayAttr{},
      ValueRange{});

  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + names.tensorTypeName + " = GlobalTensor<" + elemTypeStr +
               ", " + names.shapeTypeName + ", " + names.strideTypeName +
               ", " + names.layoutConstName + ">;");
  auto gtType = emitc::OpaqueType::get(ctx, names.tensorTypeName);

  SmallVector<Value> gtArgs;
  gtArgs.push_back(ptr);
  gtArgs.push_back(shapeInstOp.getResult(0));
  gtArgs.push_back(strideInstOp.getResult(0));

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

static Value castToGMBytePointer(ConversionPatternRewriter &rewriter,
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

static Value materializeTensorViewDataPointer(
    ConversionPatternRewriter &rewriter, Location loc, Value value,
    Type sourceType) {
  auto tvTy = dyn_cast<pto::TensorViewType>(sourceType);
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

static std::string tileBufBLayoutToken(pto::TileBufConfigAttr configAttr) {
  std::string blTok = "BLayout::RowMajor";
  if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout())) {
    if (static_cast<int32_t>(blAttr.getValue()) == 1)
      blTok = "BLayout::ColMajor";
  }
  return blTok;
}

static std::string tileBufSLayoutToken(pto::TileBufConfigAttr configAttr) {
  std::string slTok = "SLayout::NoneBox";
  if (auto slAttr = dyn_cast<SLayoutAttr>(configAttr.getSLayout())) {
    int32_t slVal = static_cast<int32_t>(slAttr.getValue());
    slTok = (slVal == 1) ? "SLayout::RowMajor"
                         : (slVal == 2) ? "SLayout::ColMajor"
                                        : "SLayout::NoneBox";
  }
  return slTok;
}

static std::string tileBufPadToken(pto::TileBufConfigAttr configAttr) {
  std::string padTok = "PadValue::Null";
  if (auto padAttr = dyn_cast<PadValueAttr>(configAttr.getPad())) {
    switch (static_cast<int32_t>(padAttr.getValue())) {
    case 1:
      padTok = "PadValue::Zero";
      break;
    case 2:
      padTok = "PadValue::Max";
      break;
    case 3:
      padTok = "PadValue::Min";
      break;
    default:
      padTok = "PadValue::Null";
      break;
    }
  }
  return padTok;
}

static pto::BLayout getTileBufBLayoutValue(pto::TileBufConfigAttr configAttr) {
  if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout()))
    return blAttr.getValue();
  return pto::BLayout::RowMajor;
}

static int64_t renderTileTemplateDim(int64_t rawDim, Type elemTy,
                                     pto::BLayout blayout, int dimIdx) {
  assert(dimIdx >= 0 && dimIdx < 2 &&
         "renderTileTemplateDim expects a rank-2 rows/cols dimension index");
  if (rawDim == ShapedType::kDynamic)
    return rawDim;
  if (!pto::isPTOFloat4PackedType(elemTy))
    return rawDim;
  int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
  return dimIdx == packedDim ? rawDim * 2 : rawDim;
}

static FailureOr<Value> buildAsyncScratchTileValue(
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
  if (!memTy.hasStaticShape() || shape.empty() || shape.size() > 2)
    return failure();

  int64_t rows = shape.size() == 1 ? 1 : shape[0];
  int64_t cols = shape.size() == 1 ? shape[0] : shape[1];

  auto *ctx = rewriter.getContext();
  pto::TileBufConfigAttr configAttr = pto::TileBufConfigAttr::getDefault(ctx);
  if (auto bind = originalScratch.getDefiningOp<pto::BindTileOp>()) {
    configAttr = bind.getConfig();
  } else if (auto cast = originalScratch.getDefiningOp<pto::PointerCastOp>()) {
    if (auto config = cast.getConfig())
      configAttr = *config;
  }

  int32_t fractal = 512;
  if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
    fractal = frAttr.getInt();

  Type elemTy = memTy.getElementType();
  pto::BLayout blayout = getTileBufBLayoutValue(configAttr);
  int64_t templateRows = renderTileTemplateDim(rows, elemTy, blayout, 0);
  int64_t templateCols = renderTileTemplateDim(cols, elemTy, blayout, 1);
  std::string elemTypeStr = getEmitCScalarTypeToken(elemTy);
  std::string tileTypeStr =
      "Tile<TileType::Vec, " + elemTypeStr + ", " +
      std::to_string(templateRows) + ", " + std::to_string(templateCols) +
      ", " + tileBufBLayoutToken(configAttr) + ", " +
      std::to_string(templateRows) + ", " + std::to_string(templateCols) +
      ", " + tileBufSLayoutToken(configAttr) + ", " +
      std::to_string(fractal) + ", " + tileBufPadToken(configAttr) + ">";

  Value tile = rewriter
                   .create<emitc::VariableOp>(
                       loc, emitc::OpaqueType::get(ctx, tileTypeStr),
                       emitc::OpaqueAttr::get(ctx, ""))
                   .getResult();
  auto addr = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
  Value scratchAddr =
      rewriter
          .create<emitc::CallOpaqueOp>(loc, emitc::OpaqueType::get(ctx, "uint64_t"),
                                       "reinterpret_cast", ArrayAttr{}, addr,
                                       ValueRange{scratch})
          .getResult(0);
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                       ArrayAttr{}, ArrayAttr{},
                                       ValueRange{tile, scratchAddr});
  return tile;
}

static FailureOr<Value> buildSyncAllWorkspaceTileValue(
    ConversionPatternRewriter &rewriter, Location loc, Value originalWorkspace,
    Value emittedWorkspace) {
  Value workspace = peelUnrealized(emittedWorkspace);
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(workspace.getType())) {
    StringRef typeStr = opaqueTy.getValue();
    if (typeStr.contains("Tile<") || typeStr.contains("ConvTile<"))
      return workspace;
  }

  auto memTy = dyn_cast<MemRefType>(originalWorkspace.getType());
  if (!memTy)
    return failure();
  if (!memTy.hasStaticShape())
    return failure();

  ArrayRef<int64_t> rawShape = memTy.getShape();
  if (rawShape.empty() || rawShape.size() > 2)
    return failure();

  int64_t rows = rawShape.size() == 1 ? 1 : rawShape[0];
  int64_t cols = rawShape.size() == 1 ? rawShape[0] : rawShape[1];
  SmallVector<int64_t, 2> shape{rows, cols};
  SmallVector<int64_t, 2> validShape{rows, cols};

  auto *ctx = rewriter.getContext();
  pto::TileBufConfigAttr configAttr = pto::TileBufConfigAttr::getDefault(ctx);
  if (auto bind = originalWorkspace.getDefiningOp<pto::BindTileOp>()) {
    configAttr = bind.getConfig();
  } else if (auto cast = originalWorkspace.getDefiningOp<pto::PointerCastOp>()) {
    if (auto config = cast.getConfig())
      configAttr = *config;
  }

  Attribute memorySpace = memTy.getMemorySpace();
  if (!memorySpace)
    return failure();

  auto tileTy = pto::TileBufType::get(ctx, shape, memTy.getElementType(),
                                      memorySpace, validShape, configAttr);
  auto tileTypeString = getEmitCTileTypeString(tileTy);
  if (!tileTypeString)
    return failure();

  auto tileEmitTy = emitc::OpaqueType::get(ctx, *tileTypeString);
  Value tile = rewriter
                   .create<emitc::VariableOp>(loc, tileEmitTy,
                                              emitc::OpaqueAttr::get(ctx, ""))
                   .getResult();

  Value rawPtr = workspace;
  auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
  if (isSetFFTsPointerLikeType(rawPtr.getType())) {
    auto rcU64 =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
    rawPtr = rewriter
                 .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                              ArrayAttr{}, rcU64,
                                              ValueRange{rawPtr})
                 .getResult(0);
  } else if (rawPtr.getType() != u64Ty) {
    rawPtr = rewriter.create<emitc::CastOp>(loc, u64Ty, rawPtr).getResult();
  }

  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                       ArrayAttr{}, ArrayAttr{},
                                       ValueRange{tile, rawPtr});
  return tile;
}

//===----------------------------------------------------------------------===//
// pto.pointer_cast lowering
//===----------------------------------------------------------------------===
struct PointerCastConversion : public OpConversionPattern<pto::PointerCastOp> {
  static bool getIndexConst(Value v, int64_t &out) {
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue())) {
        out = ia.getValue().getSExtValue();
        return true;
      }
    }
    return false;
  }

  using OpConversionPattern<pto::PointerCastOp>::OpConversionPattern;

  enum class TileRole { Vec, Mat, Left, Right, Acc, Bias, Scaling };

  static void collectUserOpsThroughCasts(Value v, SmallVectorImpl<Operation *> &out) {
    for (Operation *u : v.getUsers()) {
      if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(u)) {
        for (Value r : castOp.getResults())
          collectUserOpsThroughCasts(r, out);
        continue;
      }
      out.push_back(u);
    }
  }

  static Value peelUnrealized(Value v) {
    while (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>()) {
      v = castOp.getOperand(0);
    }
    return v;
  }

  static TileRole inferRole(pto::PointerCastOp op) {
    // 1. 优先检查 AddressSpace
    if (auto memRefTy = dyn_cast<MemRefType>(op.getType())) {
      Attribute memorySpace = memRefTy.getMemorySpace();
      if (auto ptoAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace)) {
        switch (ptoAttr.getAddressSpace()) {
          case pto::AddressSpace::LEFT:  return TileRole::Left;
          case pto::AddressSpace::RIGHT: return TileRole::Right;
          case pto::AddressSpace::ACC:   return TileRole::Acc;
          case pto::AddressSpace::BIAS:  return TileRole::Bias; 
          case pto::AddressSpace::MAT:   return TileRole::Mat;
          case pto::AddressSpace::SCALING: return TileRole::Scaling;
          default: break; 
        }
      }
    }

    // 2. 通过 Usage 推导 (Fallback)
    SmallVector<Operation *, 8> users;
    collectUserOpsThroughCasts(op.getResult(), users);

    for (Operation *user : users) {
      if (auto mm = dyn_cast<pto::TMatmulOp>(user)) {
        if (mm.getDst() && peelUnrealized(mm.getDst()) == op.getResult()) return TileRole::Acc;
        if (peelUnrealized(mm.getLhs()) == op.getResult()) return TileRole::Left;
        if (peelUnrealized(mm.getRhs()) == op.getResult()) return TileRole::Right;
      }
      if (auto mmacc = dyn_cast<pto::TMatmulAccOp>(user)) {
        if (mmacc.getDst() && peelUnrealized(mmacc.getDst()) == op.getResult()) return TileRole::Acc;
        if (peelUnrealized(mmacc.getAccIn()) == op.getResult()) return TileRole::Acc;
        if (peelUnrealized(mmacc.getLhs()) == op.getResult()) return TileRole::Left;
        if (peelUnrealized(mmacc.getRhs()) == op.getResult()) return TileRole::Right;
      }
    }

    return TileRole::Vec;
  }

  // [新增] 辅助函数：判断 Value 是否源自 arith.constant
  static bool isConstant(Value v, int64_t &outVal) {
    if (!v) return false;
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
       if (auto attr = dyn_cast<IntegerAttr>(cst.getValue())) {
           outVal = attr.getInt();
           return true;
       }
    }
    return false;
  }

  LogicalResult matchAndRewrite(pto::PointerCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto selfType = mlir::cast<MemRefType>(op.getType());
    ArrayRef<int64_t> shape = selfType.getShape();
    Type elemType = selfType.getElementType();
    
    // 1. 推导 Tile Role
    TileRole role = inferRole(op);

    // 2. 类型字符串生成 (elemTypeStr, dimStr)
    std::string elemTypeStr = getEmitCScalarTypeToken(elemType);

    std::string dimStr;
    pto::BLayout blayout = pto::BLayout::RowMajor;
    auto dimToString = [&](int64_t dim, const char *symbol,
                           int dimIdx) -> std::string {
        if (dim == ShapedType::kDynamic)
          return std::string(symbol);
        return std::to_string(renderTileTemplateDim(dim, elemType, blayout,
                                                    dimIdx));
    };

    // 3. Role Token
    const char *roleTok = "TileType::Vec";
    switch (role) {
      case TileRole::Left:  roleTok = "TileType::Left"; break;
      case TileRole::Right: roleTok = "TileType::Right"; break;
      case TileRole::Acc:   roleTok = "TileType::Acc"; break;
      case TileRole::Bias:  roleTok = "TileType::Bias"; break;
      case TileRole::Mat:   roleTok = "TileType::Mat"; break;
      case TileRole::Vec:   roleTok = "TileType::Vec"; break;
      case TileRole::Scaling: roleTok = "TileType::Scaling"; break;
    }

    // 4. Config & Layout (support BLayoutAttr/SLayoutAttr/PadValueAttr after namespace change)
    std::string layoutParams = "BLayout::RowMajor";
    std::string extraParams = "";
    if (auto configOpt = op.getConfig()) {
        auto config = *configOpt;
        int32_t blVal = 0;
        if (auto attr = dyn_cast<BLayoutAttr>(config.getBLayout()))
            blVal = static_cast<int32_t>(attr.getValue());
 
        if (blVal == 1) layoutParams = "BLayout::ColMajor";
        blayout = blVal == 1 ? pto::BLayout::ColMajor : pto::BLayout::RowMajor;

        int32_t slVal = 0;
        if (auto attr = dyn_cast<SLayoutAttr>(config.getSLayout()))
            slVal = static_cast<int32_t>(attr.getValue());

        std::string slStr = (slVal == 1) ? "SLayout::RowMajor" : (slVal == 2) ? "SLayout::ColMajor" : "SLayout::NoneBox";

        int32_t frVal = 0;
        if (auto attr = dyn_cast<IntegerAttr>(config.getSFractalSize())) frVal = attr.getInt();

        int32_t padVal = 0;
        if (auto attr = dyn_cast<PadValueAttr>(config.getPad()))
            padVal = static_cast<int32_t>(attr.getValue());

        std::string padStr = "PadValue::Null";
        switch (padVal) {
            case 1: padStr = "PadValue::Zero"; break;
            case 2: padStr = "PadValue::Max";  break;
            case 3: padStr = "PadValue::Min";  break;
        }

        int32_t compactVal = 0;
        if (auto attr = dyn_cast<CompactModeAttr>(config.getCompactMode()))
            compactVal = static_cast<int32_t>(attr.getValue());

        std::string compactStr = "CompactMode::Null";
        switch (compactVal) {
            case 1: compactStr = "CompactMode::Normal"; break;
            case 2: compactStr = "CompactMode::RowPlusOne"; break;
        }

        if (!slStr.empty()) {
            extraParams += ", " + slStr + ", " + std::to_string(frVal) + ", " +
                           padStr + ", " + compactStr;
        }
    } else {
        extraParams = ", SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null";
    }

    if (role == TileRole::Left)
      dimStr = dimToString(shape[0], "M", 0) + ", " +
               dimToString(shape[1], "K", 1);
    else if (role == TileRole::Right)
      dimStr = dimToString(shape[0], "K", 0) + ", " +
               dimToString(shape[1], "N", 1);
    else if (role == TileRole::Bias)
      dimStr = "1, " + dimToString(shape[1], "N", 1);
    else
      dimStr = dimToString(shape[0], "M", 0) + ", " +
               dimToString(shape[1], "N", 1);

    // [核心修改] Valid Dims 处理逻辑 (支持混合静态/动态)
    std::string vrowTok, vcolTok;
    bool useConstructor = false;

    bool rowIsDynamic = false;
    bool colIsDynamic = false;

    SmallVector<Value> constructorArgs;

    Value vRow = op.getValidRow();
    Value vCol = op.getValidCol();
    Value vRowEmitC = adaptor.getValidRow();
    Value vColEmitC = adaptor.getValidCol();
    bool forceDynamicValid = op->hasAttr(kForceDynamicValidShapeAttrName);

    int64_t cRow = 0, cCol = 0;
    bool rowIsConst = vRow && isConstant(vRow, cRow);
    bool colIsConst = vCol && isConstant(vCol, cCol);

    auto makeCtorDimValue = [&](Value emitted, int64_t fallback) -> Value {
      if (emitted)
        return emitted;
      return makeEmitCIntConstant(
          rewriter, loc, emitc::OpaqueType::get(ctx, "int32_t"), fallback);
    };
    auto maybeScaleDynamicValid = [&](Value emitted, int dimIdx) -> Value {
      if (!emitted || !pto::isPTOFloat4PackedType(elemType))
        return emitted;
      int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
      if (dimIdx != packedDim)
        return emitted;
      auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
      Value two = makeEmitCIntConstant(rewriter, loc, i32Ty, 2);
      return rewriter.create<emitc::MulOp>(loc, i32Ty, emitted, two).getResult();
    };

    if (forceDynamicValid) {
      vrowTok = "-1";
      vcolTok = "-1";
      useConstructor = true;
      constructorArgs.push_back(
          makeCtorDimValue(maybeScaleDynamicValid(vRowEmitC, 0),
                           renderTileTemplateDim(rowIsConst ? cRow : shape[0],
                                                 elemType, blayout, 0)));
      constructorArgs.push_back(
          makeCtorDimValue(maybeScaleDynamicValid(vColEmitC, 1),
                           renderTileTemplateDim(colIsConst ? cCol : shape[1],
                                                 elemType, blayout, 1)));
    } else {
      if (rowIsConst) {
        vrowTok = std::to_string(
            renderTileTemplateDim(cRow, elemType, blayout, 0));
      } else if (vRow) {
        vrowTok = "-1";
        rowIsDynamic = true;
        useConstructor = true;
      } else {
        vrowTok = std::to_string(
            renderTileTemplateDim(shape[0], elemType, blayout, 0));
      }

      if (colIsConst) {
        vcolTok = std::to_string(
            renderTileTemplateDim(cCol, elemType, blayout, 1));
      } else if (vCol) {
        vcolTok = "-1";
        colIsDynamic = true;
        useConstructor = true;
      } else {
        vcolTok = std::to_string(
            renderTileTemplateDim(shape[1], elemType, blayout, 1));
      }

      if (useConstructor) {
        if (rowIsDynamic && vRowEmitC)
          constructorArgs.push_back(maybeScaleDynamicValid(vRowEmitC, 0));
        if (colIsDynamic && vColEmitC)
          constructorArgs.push_back(maybeScaleDynamicValid(vColEmitC, 1));
      }
    }

    // 5. 生成 Tile 类型字符串
    std::string tileTypeStr =
      std::string("Tile<") + roleTok + ", " + elemTypeStr + ", " + dimStr + ", " +
      layoutParams + ", " + vrowTok + ", " + vcolTok + extraParams + ">";

    auto tileType = emitc::OpaqueType::get(ctx, tileTypeStr);
    Value resultValue;

    if (useConstructor) {
        // 使用 CallOpaqueOp 生成构造函数调用 (Tile v = Tile(...))
        auto ctorOp = rewriter.create<emitc::CallOpaqueOp>(
            loc, 
            tileType,        // Result Type
            tileTypeStr,     // Callee Name (类名)
            ArrayAttr{},     // args
            ArrayAttr{},     // template_args
            ValueRange(constructorArgs) // operands
        );
        resultValue = ctorOp.getResult(0);
    } else {
        // 静态情况 (Tile v;)
        auto varOp = rewriter.create<emitc::VariableOp>(
            loc, 
            tileType, 
            emitc::OpaqueAttr::get(ctx, "")
        );
        resultValue = varOp.getResult();
    }

    // TASSIGN: pto-isa expects an integral address.
    Value addr = adaptor.getAddrs()[0];
    if (isa<emitc::PointerType>(addr.getType()) ||
        (isa<emitc::OpaqueType>(addr.getType()) &&
         cast<emitc::OpaqueType>(addr.getType()).getValue().ends_with("*"))) {
      auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
      auto rcU64 = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
      addr = rewriter.create<emitc::CallOpaqueOp>(
                 loc, u64Ty, "reinterpret_cast",
                 /*args=*/ArrayAttr{}, /*templateArgs=*/rcU64,
                 /*operands=*/ValueRange{addr})
                 .getResult(0);
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TASSIGN",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{resultValue, addr});

    rewriter.replaceOp(op, resultValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.load_dps / pto.store_dps lowering (FIX: keep optional result)
//===----------------------------------------------------------------------===

struct PTOTLoadToTLOAD : public OpConversionPattern<pto::TLoadOp> {
  using OpConversionPattern<pto::TLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.tload");

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value srcArg = src;
    if (auto srcMrTy = dyn_cast<MemRefType>(op.getSrc().getType())) {
      bool isGlobal = true;
      if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(srcMrTy.getMemorySpace())) {
        auto as = asAttr.getAddressSpace();
        isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
      }
      if (isGlobal) {
        if (Value gt = buildGlobalTensorFromMemref(rewriter, op.getLoc(), src, srcMrTy,
                                                  op.getOperation()))
          srcArg = gt;
      }
    }

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TLOAD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, srcArg});

    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct PTOTPrefetchToTPREFETCH : public OpConversionPattern<pto::TPrefetchOp> {
  using OpConversionPattern<pto::TPrefetchOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPrefetchOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.tprefetch");

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value srcArg = src;
    if (auto srcMrTy = dyn_cast<MemRefType>(op.getSrc().getType())) {
      bool isGlobal = true;
      if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(srcMrTy.getMemorySpace())) {
        auto as = asAttr.getAddressSpace();
        isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
      }
      if (isGlobal) {
        if (Value gt = buildGlobalTensorFromMemref(rewriter, op.getLoc(), src, srcMrTy,
                                                  op.getOperation()))
          srcArg = gt;
      }
    }

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TPREFETCH",
        ArrayAttr{}, ArrayAttr{}, ValueRange{dst, srcArg});
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTPrefetchAsyncToEmitC
    : public OpConversionPattern<pto::TPrefetchAsyncOp> {
  using OpConversionPattern<pto::TPrefetchAsyncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPrefetchAsyncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());
    Value srcArg = src;
    if (!isEmitCGlobalTensorLikeType(srcArg.getType())) {
      auto srcMrTy = dyn_cast<MemRefType>(op.getSrc().getType());
      if (!srcMrTy)
        return rewriter.notifyMatchFailure(
            op, "expected src to lower to GlobalTensor or memref");
      srcArg = buildGlobalTensorFromMemref(rewriter, op.getLoc(), src, srcMrTy,
                                           op.getSrc().getDefiningOp()
                                               ? op.getSrc().getDefiningOp()
                                               : op.getOperation());
    }
    if (!srcArg)
      return rewriter.notifyMatchFailure(op,
                                         "failed to build GlobalTensor src");

    Value prefetchCtx = peelUnrealized(adaptor.getCtx());

    Type eventTy = getTypeConverter()->convertType(op.getEvent().getType());
    if (!eventTy)
      return rewriter.notifyMatchFailure(
          op, "failed to convert tprefetch_async result type");

    Value event = rewriter
                      .create<emitc::CallOpaqueOp>(
                          op.getLoc(), TypeRange{eventTy}, "TPREFETCH_ASYNC",
                          ArrayAttr{}, ArrayAttr{},
                          ValueRange{srcArg, prefetchCtx})
                      .getResult(0);

    rewriter.replaceOp(op, ValueRange{event});
    return success();
  }
};

struct PTOMakePrefetchAsyncContextToEmitC
    : public OpConversionPattern<pto::MakePrefetchAsyncContextOp> {
  using OpConversionPattern<pto::MakePrefetchAsyncContextOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MakePrefetchAsyncContextOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type ctxTy = getTypeConverter()->convertType(op.getCtx().getType());
    if (!ctxTy)
      return rewriter.notifyMatchFailure(
          op, "failed to convert make_prefetch_async_context result type");

    Value workspace = peelUnrealized(adaptor.getWorkspace());
    workspace = castToGMBytePointer(rewriter, op.getLoc(), workspace);

    Value ctx = rewriter
                    .create<emitc::CallOpaqueOp>(
                        op.getLoc(), TypeRange{ctxTy}, "pto::PrefetchAsyncContext",
                        ArrayAttr{}, ArrayAttr{}, ValueRange{workspace})
                    .getResult(0);

    rewriter.replaceOp(op, ValueRange{ctx});
    return success();
  }
};

struct PTOGetPrefetchAsyncSessionToEmitC
    : public OpConversionPattern<pto::GetPrefetchAsyncSessionOp> {
  using OpConversionPattern<pto::GetPrefetchAsyncSessionOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::GetPrefetchAsyncSessionOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type sessionTy = getTypeConverter()->convertType(op.getSession().getType());
    if (!sessionTy)
      return rewriter.notifyMatchFailure(
          op, "failed to convert get_prefetch_async_session result type");

    Value ctx = peelUnrealized(adaptor.getCtx());
    Value session = rewriter
                        .create<emitc::CallOpaqueOp>(
                            op.getLoc(), TypeRange{sessionTy},
                            "PTOAS__PREFETCH_CTX_SESSION", ArrayAttr{},
                            ArrayAttr{}, ValueRange{ctx})
                        .getResult(0);

    rewriter.replaceOp(op, ValueRange{session});
    return success();
  }
};

struct PTOTStoreToTSTORE : public OpConversionPattern<pto::TStoreOp> {
  using OpConversionPattern<pto::TStoreOp>::OpConversionPattern;

  static std::string stPhaseTok(pto::STPhase phase) {
    switch (phase) {
      case pto::STPhase::Unspecified: return "STPhase::Unspecified";
      case pto::STPhase::Partial: return "STPhase::Partial";
      case pto::STPhase::Final: return "STPhase::Final";
    }
    return "STPhase::Unspecified";
  }

  static std::string atomicTypeTok(pto::AtomicType atomicType) {
    switch (atomicType) {
      case pto::AtomicType::AtomicNone: return "AtomicType::AtomicNone";
      case pto::AtomicType::AtomicAdd: return "AtomicType::AtomicAdd";
    }
    return "AtomicType::AtomicNone";
  }

  static std::string reluPreModeTok(pto::ReluPreMode reluPreMode) {
    switch (reluPreMode) {
      case pto::ReluPreMode::NoRelu: return "ReluPreMode::NoRelu";
      case pto::ReluPreMode::NormalRelu: return "ReluPreMode::NormalRelu";
      case pto::ReluPreMode::ScalarRelu: return "ReluPreMode::ScalarRelu";
      case pto::ReluPreMode::VectorRelu: return "ReluPreMode::VectorRelu";
      case pto::ReluPreMode::Pwl: return "ReluPreMode::Pwl";
    }
    return "ReluPreMode::NoRelu";
  }

  LogicalResult matchAndRewrite(pto::TStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.tstore");

    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value preQuantScalar;
    if (op.getPreQuantScalar())
      preQuantScalar = peelUnrealized(adaptor.getPreQuantScalar());
    Value dstArg = dst;
    if (auto dstMrTy = dyn_cast<MemRefType>(op.getDst().getType())) {
      bool isGlobal = true;
      if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(dstMrTy.getMemorySpace())) {
        auto as = asAttr.getAddressSpace();
        isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
      }
      if (isGlobal) {
        if (Value gt = buildGlobalTensorFromMemref(rewriter, op.getLoc(), dst, dstMrTy,
                                                  op.getOperation()))
          dstArg = gt;
      }
    }

    const auto phase = op.getStPhase();
    const auto atomicType = op.getAtomicType();
    const auto reluPreMode = op.getReluPreMode();
    const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);
    const bool phaseNonDefault = phase != pto::STPhase::Unspecified;
    const bool atomicNonDefault = atomicType != pto::AtomicType::AtomicNone;
    const bool reluNonDefault = reluPreMode != pto::ReluPreMode::NoRelu;

    auto getOpaqueTok = [&](Value v, StringRef name) -> FailureOr<std::string> {
      if (auto ot = mlir::dyn_cast<emitc::OpaqueType>(v.getType()))
        return ot.getValue().str();
      return rewriter.notifyMatchFailure(op, (name + " must be emitc::OpaqueType").str());
    };

    ArrayAttr targs;
    // Map op attributes/operands to the exact TSTORE overload family:
    //  1) TSTORE(dst, src)
    //  2) TSTORE<Phase>(dst, src)
    //  3) TSTORE<TileData, GlobalData, AtomicType>(dst, src)
    //  4) TSTORE<Phase, TileData, GlobalData, AtomicType>(dst, src)
    //  5) TSTORE<TileData, GlobalData, AtomicType, ReluPreMode>(dst, src)
    //  6) TSTORE<Phase, TileData, GlobalData, AtomicType, ReluPreMode>(dst, src)
    //  7) TSTORE<TileData, GlobalData, AtomicType, ReluPreMode>(dst, src, preQuant)
    //  8) TSTORE<Phase, TileData, GlobalData, AtomicType, ReluPreMode>(dst, src, preQuant)
    if (!hasPreQuantScalar && !reluNonDefault && !atomicNonDefault) {
      if (phaseNonDefault) {
        targs = rewriter.getArrayAttr({
            emitc::OpaqueAttr::get(ctx, stPhaseTok(phase)),
        });
      } else {
        targs = ArrayAttr{};
      }
    } else {
      auto srcTokOr = getOpaqueTok(src, "src");
      auto dstTokOr = getOpaqueTok(dstArg, "dst");
      if (failed(srcTokOr) || failed(dstTokOr))
        return failure();

      // If there is no preQuant and relu stays default, emit the atomic-only
      // overloads (#3/#4) without ReluPreMode template argument.
      if (!hasPreQuantScalar && !reluNonDefault) {
        if (phaseNonDefault) {
          targs = rewriter.getArrayAttr({
              emitc::OpaqueAttr::get(ctx, stPhaseTok(phase)),
              emitc::OpaqueAttr::get(ctx, *srcTokOr),
              emitc::OpaqueAttr::get(ctx, *dstTokOr),
              emitc::OpaqueAttr::get(ctx, atomicTypeTok(atomicType)),
          });
        } else {
          targs = rewriter.getArrayAttr({
              emitc::OpaqueAttr::get(ctx, *srcTokOr),
              emitc::OpaqueAttr::get(ctx, *dstTokOr),
              emitc::OpaqueAttr::get(ctx, atomicTypeTok(atomicType)),
          });
        }
      } else {
        // Relu/preQuant families (#5/#6/#7/#8): keep AtomicType + ReluPreMode.
        if (phaseNonDefault) {
          targs = rewriter.getArrayAttr({
              emitc::OpaqueAttr::get(ctx, stPhaseTok(phase)),
              emitc::OpaqueAttr::get(ctx, *srcTokOr),
              emitc::OpaqueAttr::get(ctx, *dstTokOr),
              emitc::OpaqueAttr::get(ctx, atomicTypeTok(atomicType)),
              emitc::OpaqueAttr::get(ctx, reluPreModeTok(reluPreMode)),
          });
        } else {
          targs = rewriter.getArrayAttr({
              emitc::OpaqueAttr::get(ctx, *srcTokOr),
              emitc::OpaqueAttr::get(ctx, *dstTokOr),
              emitc::OpaqueAttr::get(ctx, atomicTypeTok(atomicType)),
              emitc::OpaqueAttr::get(ctx, reluPreModeTok(reluPreMode)),
          });
        }
      }
    }

    SmallVector<Value, 3> operands{dstArg, src};
    if (hasPreQuantScalar)
      operands.push_back(preQuantScalar);

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSTORE",
        /*args=*/ArrayAttr{}, /*templateArgs=*/targs,
        /*operands=*/operands);

    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.matmul_dps lowering (Simplified: No internal copy/sync)
//===----------------------------------------------------------------------===//
//
// Render `pto.tmatmul` as one of three forms depending on the optional
// `acc_phase` attribute:
//   * absent / Unspecified  -> `TMATMUL(dst, lhs, rhs)`
//   * Partial               -> `TMATMUL<pto::AccPhase::Partial>(dst, lhs, rhs)`
//   * Final                 -> `TMATMUL<pto::AccPhase::Final>(dst, lhs, rhs)`
// The Unspecified default keeps backward compatibility with all upstream IR
// that does not yet emit an explicit phase attribute.
static ArrayAttr buildAccPhaseTemplateArgs(ConversionPatternRewriter &rewriter,
                                           pto::AccPhase phase) {
  StringRef tmpl;
  switch (phase) {
  case pto::AccPhase::Unspecified:
    return ArrayAttr{};
  case pto::AccPhase::Partial:
    tmpl = "pto::AccPhase::Partial";
    break;
  case pto::AccPhase::Final:
    tmpl = "pto::AccPhase::Final";
    break;
  }
  if (tmpl.empty())
    return ArrayAttr{};
  return rewriter.getArrayAttr(
      {emitc::OpaqueAttr::get(rewriter.getContext(), tmpl)});
}

struct PTOTMatmulToTMATMUL : public OpConversionPattern<pto::TMatmulOp> {
  using OpConversionPattern<pto::TMatmulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取操作数 (剥离 Cast)
    Value lhs = peelUnrealized(adaptor.getLhs()); // A (Left)
    Value rhs = peelUnrealized(adaptor.getRhs()); // B (Right)
    Value dst = peelUnrealized(adaptor.getDst()); // C (Acc)

    // 2. 根据 acc_phase 属性决定是否生成 TMATMUL<AccPhase::Final/Partial>(...)
    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TMATMUL",
        /*args=*/ArrayAttr{}, /*template_args=*/templateArgs,
        ValueRange{dst, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.tgemv lowering
//===----------------------------------------------------------------------===//
struct PTOTGemvToTGEMV : public OpConversionPattern<pto::TGemvOp> {
  using OpConversionPattern<pto::TGemvOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取操作数 (剥离 Cast)
    Value lhs = peelUnrealized(adaptor.getLhs()); // A (Matrix)
    Value rhs = peelUnrealized(adaptor.getRhs()); // B (Vector)
    Value dst = peelUnrealized(adaptor.getDst()); // C (Result)

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TGEMV",
        /*args=*/ArrayAttr{}, /*template_args=*/templateArgs,
        ValueRange{dst, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.tgemv.acc lowering
//===----------------------------------------------------------------------===//
struct PTOTGemvAccToTGEMVACC : public OpConversionPattern<pto::TGemvAccOp> {
  using OpConversionPattern<pto::TGemvAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) for pto.tgemv.acc");

    // 1. 获取操作数
    Value accIn = peelUnrealized(adaptor.getAccIn()); // AccOld
    Value lhs   = peelUnrealized(adaptor.getLhs());   // A (Matrix)
    Value rhs   = peelUnrealized(adaptor.getRhs());   // B (Vector)
    Value dst   = peelUnrealized(adaptor.getDst());   // AccNew

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TGEMV_ACC",
        /*args=*/ArrayAttr{}, /*template_args=*/templateArgs,
        ValueRange{dst, accIn, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.matmul_acc_dps lowering (Simplified: No internal copy/sync)
//===----------------------------------------------------------------------===//
struct PTOTMatmulAccToTMATMULACC : public OpConversionPattern<pto::TMatmulAccOp> {
  using OpConversionPattern<pto::TMatmulAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) for pto.tmatmul.acc");

    // 1. 获取操作数
    Value accIn = peelUnrealized(adaptor.getAccIn()); // AccOld
    Value lhs   = peelUnrealized(adaptor.getLhs());   // A (Left)
    Value rhs   = peelUnrealized(adaptor.getRhs());   // B (Right)
    Value dst   = peelUnrealized(adaptor.getDst());   // AccNew

    // 2. 根据 acc_phase 属性决定是否生成 TMATMUL_ACC<AccPhase::Final/Partial>(...)
    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TMATMUL_ACC",
        /*args=*/ArrayAttr{}, /*template_args=*/templateArgs,
        ValueRange{dst, accIn, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Return lowering
//===----------------------------------------------------------------------===

static constexpr llvm::StringLiteral kAutoSyncTailPendingModeAttr =
    "__pto.auto_sync_tail_mode";

struct ReturnToEmitC : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (auto emitcFunc = op->getParentOfType<emitc::FuncOp>()) {
      if (auto modeAttr =
              emitcFunc->getAttrOfType<StringAttr>(kAutoSyncTailPendingModeAttr)) {
        auto *ctx = rewriter.getContext();
        rewriter.setInsertionPoint(op);
        auto args = rewriter.getArrayAttr(
            {emitc::OpaqueAttr::get(ctx, modeAttr.getValue())});
        rewriter.create<emitc::CallOpaqueOp>(
            op.getLoc(), TypeRange{}, "ptoas_auto_sync_tail",
            args, ArrayAttr{}, ValueRange{});
      }
    }

    auto vals = adaptor.getOperands();
    if (vals.empty()) {
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(op, Value{});
      return success();
    }
    if (vals.size() == 1) {
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(op, vals[0]);
      return success();
    }
    return rewriter.notifyMatchFailure(op, "EmitC cannot return multiple values");
  }
};

struct CallToEmitC : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "EmitC cannot lower calls with multiple results");

    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "failed to convert call result types");

    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, op.getCalleeAttr(),
                                               resultTypes,
                                               adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Sync lowering
//===----------------------------------------------------------------------===

static constexpr llvm::StringLiteral kAutoSyncTailBarrierAttr =
    "pto.auto_sync_tail_barrier";
static constexpr llvm::StringLiteral kAutoSyncTailHintAttr =
    "pto.auto_sync_tail_hint";
static constexpr llvm::StringLiteral kAutoSyncTailPolicyBarrierAll =
    "barrier_all";
static constexpr llvm::StringLiteral kAutoSyncTailPolicyMte3ToSEvent0 =
    "setwait_mte3_to_s_event0";
static constexpr llvm::StringLiteral kAutoSyncTailModeBarrierAllToken =
    "PTOAutoSyncTailMode::kBarrierAll";
static constexpr llvm::StringLiteral kAutoSyncTailModeMte3ToSEvent0Token =
    "PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0";

static std::string getAutoSyncTailModeToken(Operation *op) {
  if (op) {
    if (auto hintAttr = op->getAttrOfType<StringAttr>(kAutoSyncTailHintAttr)) {
      if (hintAttr.getValue() == kAutoSyncTailPolicyBarrierAll)
        return kAutoSyncTailModeBarrierAllToken.str();
      if (hintAttr.getValue() == kAutoSyncTailPolicyMte3ToSEvent0)
        return kAutoSyncTailModeMte3ToSEvent0Token.str();
    }
  }

  auto func = op ? op->getParentOfType<func::FuncOp>() : func::FuncOp();
  if (!func)
    return kAutoSyncTailModeBarrierAllToken.str();

  auto hintAttr = func->getAttrOfType<StringAttr>(kAutoSyncTailHintAttr);
  if (!hintAttr)
    return kAutoSyncTailModeBarrierAllToken.str();

  if (hintAttr.getValue() == kAutoSyncTailPolicyBarrierAll)
    return kAutoSyncTailModeBarrierAllToken.str();
  if (hintAttr.getValue() == kAutoSyncTailPolicyMte3ToSEvent0)
    return kAutoSyncTailModeMte3ToSEvent0Token.str();

  // Fallback to the conservative behavior when seeing unknown policies.
  return kAutoSyncTailModeBarrierAllToken.str();
}

[[maybe_unused]] static std::string getPipeName(pto::PIPE pipe) {
  switch (pipe) {
    case pto::PIPE::PIPE_S: return "PIPE_S";
    case pto::PIPE::PIPE_V: return "PIPE_V";
    case pto::PIPE::PIPE_M: return "PIPE_M";
    case pto::PIPE::PIPE_MTE1: return "PIPE_MTE1";
    case pto::PIPE::PIPE_MTE2: return "PIPE_MTE2";
    case pto::PIPE::PIPE_MTE3: return "PIPE_MTE3";
    case pto::PIPE::PIPE_ALL: return "PIPE_ALL";
    case pto::PIPE::PIPE_MTE4: return "PIPE_MTE4";
    case pto::PIPE::PIPE_MTE5: return "PIPE_MTE5";
    case pto::PIPE::PIPE_V2: return "PIPE_V2";
    case pto::PIPE::PIPE_FIX: return "PIPE_FIX";
    case pto::PIPE::VIRTUAL_PIPE_MTE2_L1A: return "VIRTUAL_PIPE_MTE2_L1A";
    case pto::PIPE::VIRTUAL_PIPE_MTE2_L1B: return "VIRTUAL_PIPE_MTE2_L1B";
    // 默认回退
    default: return "PIPE_ALL"; 
  }
}

//===----------------------------------------------------------------------===//
// pto.barrier lowering -> pipe_barrier(...)
//===----------------------------------------------------------------------===//
struct PTOBarrierToEmitC : public OpConversionPattern<pto::BarrierOp> {
  using OpConversionPattern<pto::BarrierOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::BarrierOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op->hasAttr(kAutoSyncTailBarrierAttr)) {
      auto modeAttr = rewriter.getStringAttr(getAutoSyncTailModeToken(op));
      if (auto emitcFunc = op->getParentOfType<emitc::FuncOp>()) {
        emitcFunc->setAttr(kAutoSyncTailPendingModeAttr, modeAttr);
      } else if (auto funcOp = op->getParentOfType<func::FuncOp>()) {
        funcOp->setAttr(kAutoSyncTailPendingModeAttr, modeAttr);
      }
      rewriter.eraseOp(op);
      return success();
    }

    // [FIX] op.getPipe() returns PipeAttr. 
    // We must call .getPipe() on the attribute to get the actual Enum value.
    pto::PIPE pipeEnum = op.getPipe().getPipe();

    // Convert Enum to String (e.g., PIPE_ALL -> "PIPE_ALL")
    std::string pipeStr = pto::stringifyPIPE(pipeEnum).str();
    auto *ctx = rewriter.getContext();

    auto args = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeStr)
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, 
        TypeRange{},        // void return
        "pipe_barrier",     // function name
        args,               // arguments
        ArrayAttr{},        // template args
        ValueRange{}        // operands
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Sync lowering (robust for bracket form pto.set_flag[...] / pto.wait_flag[...])
// Replace your PTOSyncToRuntimeCall with the code below.
//===----------------------------------------------------------------------===//

static bool tryConvertPipeAttrToToken(Attribute attr, std::string &token) {
  if (!attr)
    return false;
  if (auto pipe = dyn_cast<mlir::pto::PipeAttr>(attr)) {
    token = mlir::pto::stringifyPIPE(pipe.getPipe()).str();
    return true;
  }
  if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
    token = stringAttr.getValue().str();
    return true;
  }
  return false;
}

static bool tryConvertEventAttrToToken(Attribute attr, std::string &token) {
  if (!attr)
    return false;
  if (auto event = dyn_cast<mlir::pto::EventAttr>(attr)) {
    token = mlir::pto::stringifyEVENT(event.getEvent()).str();
    return true;
  }
  if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
    token = stringAttr.getValue().str();
    return true;
  }
  return false;
}

static bool tryAssignSyncTokens(Attribute srcAttr, Attribute dstAttr,
                                Attribute evtAttr, std::string &srcTok,
                                std::string &dstTok, std::string &evtTok) {
  std::string localSrc;
  std::string localDst;
  std::string localEvt;
  if (!tryConvertPipeAttrToToken(srcAttr, localSrc) ||
      !tryConvertPipeAttrToToken(dstAttr, localDst) ||
      !tryConvertEventAttrToToken(evtAttr, localEvt)) {
    return false;
  }
  srcTok = std::move(localSrc);
  dstTok = std::move(localDst);
  evtTok = std::move(localEvt);
  return true;
}

static bool tryExtractSyncTokensFromNamedAttrs(Operation *op,
                                               StringRef srcName,
                                               StringRef dstName,
                                               StringRef evtName,
                                               std::string &srcTok,
                                               std::string &dstTok,
                                               std::string &evtTok) {
  return tryAssignSyncTokens(op->getAttr(srcName), op->getAttr(dstName),
                             op->getAttr(evtName), srcTok, dstTok, evtTok);
}

static bool tryExtractSyncTokensFromArrayAttr(Operation *op, StringRef attrName,
                                              std::string &srcTok,
                                              std::string &dstTok,
                                              std::string &evtTok) {
  auto arrayAttr = op->getAttrOfType<ArrayAttr>(attrName);
  if (!arrayAttr || arrayAttr.size() < 3)
    return false;
  return tryAssignSyncTokens(arrayAttr[0], arrayAttr[1], arrayAttr[2], srcTok,
                             dstTok, evtTok);
}

static bool tryExtractFallbackSyncTokens(Operation *op, std::string &srcTok,
                                         std::string &dstTok,
                                         std::string &evtTok) {
  SmallVector<std::string, 2> pipes;
  std::string event;
  for (NamedAttribute namedAttr : op->getAttrs()) {
    std::string token;
    if (tryConvertPipeAttrToToken(namedAttr.getValue(), token)) {
      pipes.push_back(std::move(token));
      continue;
    }
    if (event.empty() &&
        tryConvertEventAttrToToken(namedAttr.getValue(), token)) {
      event = std::move(token);
    }
  }
  if (pipes.size() < 2 || event.empty())
    return false;
  srcTok = pipes[0];
  dstTok = pipes[1];
  evtTok = event;
  return true;
}

static LogicalResult extractSyncTripletTokens(Operation *op,
                                             std::string &srcTok,
                                             std::string &dstTok,
                                             std::string &evtTok,
                                             ConversionPatternRewriter &rewriter) {
  if (tryExtractSyncTokensFromNamedAttrs(op, "src_pipe", "dst_pipe", "event_id",
                                         srcTok, dstTok, evtTok) ||
      tryExtractSyncTokensFromNamedAttrs(op, "srcPipe", "dstPipe", "eventId",
                                         srcTok, dstTok, evtTok) ||
      tryExtractSyncTokensFromNamedAttrs(op, "src", "dst", "event", srcTok,
                                         dstTok, evtTok)) {
    return success();
  }

  for (StringRef attrName : {"args", "pipes", "sync", "triplet", "attrs"}) {
    if (tryExtractSyncTokensFromArrayAttr(op, attrName, srcTok, dstTok,
                                          evtTok)) {
      return success();
    }
  }

  if (tryExtractFallbackSyncTokens(op, srcTok, dstTok, evtTok))
    return success();
  return rewriter.notifyMatchFailure(
      op, "cannot extract PIPE/PIPE/EVENT tokens from pto.{set,wait}_flag");
}
static inline std::string pipeTokFromPipeEnum(mlir::pto::PIPE p) {
  return mlir::pto::stringifyPIPE(p).str();
}
[[maybe_unused]] static inline std::string evtTokFromEventEnum(mlir::pto::EVENT e) {
  return mlir::pto::stringifyEVENT(e).str();
}
static inline std::string pipeTokFromPipeAttr(mlir::pto::PipeAttr a) {
  return mlir::pto::stringifyPIPE(a.getPipe()).str();
}
static inline std::string evtTokFromEventAttr(mlir::pto::EventAttr a) {
  return mlir::pto::stringifyEVENT(a.getEvent()).str();
}

template <typename T, typename = void>
struct HasGetSrcPipe : std::false_type {};
template <typename T>
struct HasGetSrcPipe<T, std::void_t<decltype(std::declval<T>().getSrcPipe())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetDstPipe : std::false_type {};
template <typename T>
struct HasGetDstPipe<T, std::void_t<decltype(std::declval<T>().getDstPipe())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetEventId : std::false_type {};
template <typename T>
struct HasGetEventId<T, std::void_t<decltype(std::declval<T>().getEventId())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetSrcPipeAttr : std::false_type {};
template <typename T>
struct HasGetSrcPipeAttr<T, std::void_t<decltype(std::declval<T>().getSrcPipeAttr())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetDstPipeAttr : std::false_type {};
template <typename T>
struct HasGetDstPipeAttr<T, std::void_t<decltype(std::declval<T>().getDstPipeAttr())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetEventIdAttr : std::false_type {};
template <typename T>
struct HasGetEventIdAttr<T, std::void_t<decltype(std::declval<T>().getEventIdAttr())>> : std::true_type {};

template <typename SyncOpT>
static LogicalResult extractSyncTokens(SyncOpT op,
                                      std::string &srcTok,
                                      std::string &dstTok,
                                      std::string &evtTok,
                                      ConversionPatternRewriter &rewriter) {
  if constexpr (HasGetSrcPipe<SyncOpT>::value &&
                HasGetDstPipe<SyncOpT>::value &&
                HasGetEventId<SyncOpT>::value) {
    auto s = op.getSrcPipe();
    auto d = op.getDstPipe();
    auto e = op.getEventId();

    if constexpr (std::is_same<decltype(s), mlir::pto::PIPE>::value) srcTok = pipeTokFromPipeEnum(s);
    else srcTok = pipeTokFromPipeAttr(s);

    if constexpr (std::is_same<decltype(d), mlir::pto::PIPE>::value) dstTok = pipeTokFromPipeEnum(d);
    else dstTok = pipeTokFromPipeAttr(d);

    if constexpr (std::is_same<decltype(e), mlir::pto::EVENT>::value) evtTok = evtTokFromEventEnum(e);
    else evtTok = evtTokFromEventAttr(e);

    return success();
  }

  if constexpr (HasGetSrcPipeAttr<SyncOpT>::value &&
                HasGetDstPipeAttr<SyncOpT>::value &&
                HasGetEventIdAttr<SyncOpT>::value) {
    auto s = op.getSrcPipeAttr();
    auto d = op.getDstPipeAttr();
    auto e = op.getEventIdAttr();
    srcTok = pipeTokFromPipeAttr(s);
    dstTok = pipeTokFromPipeAttr(d);
    evtTok = evtTokFromEventAttr(e);
    return success();
  }

  return extractSyncTripletTokens(op.getOperation(), srcTok, dstTok, evtTok, rewriter);
}
struct PTOSetFlagToEmitC : public OpConversionPattern<mlir::pto::SetFlagOp> {
  using OpConversionPattern<mlir::pto::SetFlagOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::SetFlagOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();

    std::string srcTok, dstTok, evtTok;
    if (failed(extractSyncTokens(op, srcTok, dstTok, evtTok, rewriter)))
      return failure();

    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, srcTok),
        emitc::OpaqueAttr::get(ctx, dstTok),
        emitc::OpaqueAttr::get(ctx, evtTok),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "set_flag",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});
    return success();
  }
};

struct PTOWaitFlagToEmitC : public OpConversionPattern<mlir::pto::WaitFlagOp> {
  using OpConversionPattern<mlir::pto::WaitFlagOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::WaitFlagOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();

    std::string srcTok, dstTok, evtTok;
    if (failed(extractSyncTokens(op, srcTok, dstTok, evtTok, rewriter)))
      return failure();

    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, srcTok),
        emitc::OpaqueAttr::get(ctx, dstTok),
        emitc::OpaqueAttr::get(ctx, evtTok),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "wait_flag",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});
    return success();
  }
};

struct PTOSyncToEmitC : public OpConversionPattern<mlir::pto::TSyncOp> {
  using OpConversionPattern<mlir::pto::TSyncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::TSyncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 4> operands;
    operands.reserve(adaptor.getEvents().size());
    for (Value event : adaptor.getEvents())
      operands.push_back(peelUnrealized(event));

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TSYNC",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(operands));
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSyncAllToEmitC : public OpConversionPattern<mlir::pto::SyncAllOp> {
  using OpConversionPattern<mlir::pto::SyncAllOp>::OpConversionPattern;

  static StringRef coreTypeTok(pto::SyncCoreType coreType) {
    switch (coreType) {
    case pto::SyncCoreType::AIVOnly:
      return "SyncCoreType::AIVOnly";
    case pto::SyncCoreType::AICOnly:
      return "SyncCoreType::AICOnly";
    case pto::SyncCoreType::Mix:
      return "SyncCoreType::Mix";
    }
    llvm_unreachable("unhandled SyncCoreType");
  }

  LogicalResult matchAndRewrite(mlir::pto::SyncAllOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto mode = op.getMode().getValue();
    auto coreType = op.getCoreType().getValue();

    auto buildGmWorkspace = [&]() -> FailureOr<Value> {
      Value gm = peelUnrealized(adaptor.getGmWorkspace());
      if (isEmitCGlobalTensorLikeType(gm.getType()))
        return gm;

      auto memTy = dyn_cast<MemRefType>(op.getGmWorkspace().getType());
      if (!memTy)
        return failure();

      Value gt = buildGlobalTensorFromMemref(rewriter, op.getLoc(), gm, memTy,
                                            op.getGmWorkspace().getDefiningOp()
                                                ? op.getGmWorkspace().getDefiningOp()
                                                : op.getOperation());
      if (!gt)
        return failure();
      return gt;
    };

    if (mode == pto::SyncAllMode::Hard) {
      std::string callee = "SYNCALL<" + coreTypeTok(coreType).str() + ">";
      rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{});
      rewriter.eraseOp(op);
      return success();
    }

    FailureOr<Value> gmWorkspace = buildGmWorkspace();
    if (failed(gmWorkspace))
      return rewriter.notifyMatchFailure(op,
                                         "failed to build gm_workspace GlobalTensor");

    auto i32Ty = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
    Value usedCores = adaptor.getUsedCores()
                          ? peelUnrealized(adaptor.getUsedCores())
                          : makeEmitCIntConstant(rewriter, op.getLoc(), i32Ty, 0);
    if (usedCores.getType() != i32Ty)
      usedCores = rewriter.create<emitc::CastOp>(op.getLoc(), i32Ty, usedCores)
                      .getResult();

    std::string callee =
        "SYNCALL<SyncAllMode::Soft, " + coreTypeTok(coreType).str() + ">";

    SmallVector<Value, 4> operands{*gmWorkspace};
    switch (coreType) {
    case pto::SyncCoreType::AIVOnly: {
      FailureOr<Value> ubWorkspace =
          buildSyncAllWorkspaceTileValue(rewriter, op.getLoc(),
                                         op.getUbWorkspace(),
                                         adaptor.getUbWorkspace());
      if (failed(ubWorkspace))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize ub_workspace tile");
      operands.push_back(*ubWorkspace);
      break;
    }
    case pto::SyncCoreType::AICOnly: {
      FailureOr<Value> l1Workspace =
          buildSyncAllWorkspaceTileValue(rewriter, op.getLoc(),
                                         op.getL1Workspace(),
                                         adaptor.getL1Workspace());
      if (failed(l1Workspace))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize l1_workspace tile");
      operands.push_back(*l1Workspace);
      break;
    }
    case pto::SyncCoreType::Mix: {
      FailureOr<Value> ubWorkspace =
          buildSyncAllWorkspaceTileValue(rewriter, op.getLoc(),
                                         op.getUbWorkspace(),
                                         adaptor.getUbWorkspace());
      FailureOr<Value> l1Workspace =
          buildSyncAllWorkspaceTileValue(rewriter, op.getLoc(),
                                         op.getL1Workspace(),
                                         adaptor.getL1Workspace());
      if (failed(ubWorkspace) || failed(l1Workspace))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize mixed syncall workspace tiles");
      operands.push_back(*ubWorkspace);
      operands.push_back(*l1Workspace);
      break;
    }
    }

    operands.push_back(usedCores);
    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange(operands));
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSyncFlagDynToEmitC : public ConversionPattern {
  PTOSyncFlagDynToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                        StringRef opName, StringRef callee)
      : ConversionPattern(typeConverter, opName, /*benefit=*/1, ctx),
        callee(callee.str()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 1)
      return rewriter.notifyMatchFailure(op, "expected exactly one dynamic event-id operand");

    auto srcAttr = op->getAttrOfType<mlir::pto::PipeAttr>("src_pipe");
    auto dstAttr = op->getAttrOfType<mlir::pto::PipeAttr>("dst_pipe");
    if (!srcAttr || !dstAttr)
      return rewriter.notifyMatchFailure(op, "missing PipeAttr src_pipe/dst_pipe attrs");

    auto *ctx = rewriter.getContext();
    std::string srcTok = pipeTokFromPipeAttr(srcAttr);
    std::string dstTok = pipeTokFromPipeAttr(dstAttr);

    Value eventVal = operands.front();
    eventVal =
        emitCCast(rewriter, op->getLoc(), emitc::OpaqueType::get(ctx, "event_t"), eventVal);

    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, srcTok),
        emitc::OpaqueAttr::get(ctx, dstTok),
        IntegerAttr::get(IndexType::get(ctx), 0),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee,
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{eventVal});
    return success();
  }

private:
  std::string callee;
};

struct PTOGetBufToEmitC : public OpConversionPattern<mlir::pto::GetBufOp> {
  using OpConversionPattern<mlir::pto::GetBufOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::GetBufOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();

    auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
    if (failed(opTypeOr))
      return rewriter.notifyMatchFailure(op, "get_buf expects pipe_event_type/sync_op_type attr");
    auto pipe = mapSyncOpTypeToPipe(*opTypeOr);
    if (!isConcreteSyncPipe(pipe))
      return rewriter.notifyMatchFailure(op, "get_buf op_type cannot map to a concrete pipe");
    std::string pipeTok = pipeTokFromPipeEnum(pipe);
    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        op.getBufIdAttr(),
        op.getModeAttr(),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "get_buf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});
    return success();
  }
};

struct PTORlsBufToEmitC : public OpConversionPattern<mlir::pto::RlsBufOp> {
  using OpConversionPattern<mlir::pto::RlsBufOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::RlsBufOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();

    auto opTypeOr = parseSyncOpTypeLikeAttr(op.getOpTypeAttr());
    if (failed(opTypeOr))
      return rewriter.notifyMatchFailure(op, "rls_buf expects pipe_event_type/sync_op_type attr");
    auto pipe = mapSyncOpTypeToPipe(*opTypeOr);
    if (!isConcreteSyncPipe(pipe))
      return rewriter.notifyMatchFailure(op, "rls_buf op_type cannot map to a concrete pipe");
    std::string pipeTok = pipeTokFromPipeEnum(pipe);
    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeTok),
        op.getBufIdAttr(),
        op.getModeAttr(),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "rls_buf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});
    return success();
  }
};

struct PTOSetFFTsToEmitC : public OpConversionPattern<mlir::pto::SetFFTsOp> {
  using OpConversionPattern<mlir::pto::SetFFTsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::SetFFTsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    Value fftsAddr = peelUnrealized(adaptor.getFfts());
    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");

    if (isSetFFTsPointerLikeType(fftsAddr.getType())) {
      auto castTyAttr =
          rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
      fftsAddr =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                           /*args=*/ArrayAttr{},
                                           /*templateArgs=*/castTyAttr,
                                           /*operands=*/ValueRange{fftsAddr})
              .getResult(0);
    } else if (fftsAddr.getType() != u64Ty) {
      fftsAddr =
          rewriter.create<emitc::CastOp>(loc, u64Ty, fftsAddr).getResult();
    }

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "set_ffts_base_addr",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{fftsAddr});
    return success();
  }
};

struct PTOSyncSetToEmitC : public OpConversionPattern<mlir::pto::SyncSetOp> {
  PTOSyncSetToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                    PTOArch targetArch)
      : OpConversionPattern<mlir::pto::SyncSetOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult
  matchAndRewrite(mlir::pto::SyncSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    IntegerAttr eventIdAttr = op.getEventIdAttr();
    Value eventIdDyn = adaptor.getEventIdDyn();
    int64_t fftsMode = 2;
    if (IntegerAttr fftsModeAttr = op.getFftsModeAttr())
      fftsMode = fftsModeAttr.getInt();

    if ((eventIdAttr != nullptr) == static_cast<bool>(eventIdDyn))
      return rewriter.notifyMatchFailure(
          op, "expects exactly one of static event_id attr or dynamic event_id operand");

    // A5 inter-core sync mirrors +16 only for cube-side producer (PIPE_FIX).
    // Vec-side producer (PIPE_MTE3) emits a single set; hardware handles the
    // subblock mapping in PTO-ISA custom flow.
    if (targetArch == PTOArch::A5) {
      pto::PIPE pipe = op.getPipe().getPipe();
      bool needsMirrorPlus16 = (pipe == pto::PIPE::PIPE_FIX);
      std::string pipeTok = pipeTokFromPipeAttr(op.getPipe());
      auto emitSet = [&](Value eventOperand, IntegerAttr eventLiteral,
                         bool isDynamic) {
        if (isDynamic) {
          auto args = rewriter.getArrayAttr({
              emitc::OpaqueAttr::get(ctx, pipeTok),
              IntegerAttr::get(IndexType::get(ctx), 0),
          });
          rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "set_intra_block",
                                               /*args=*/args,
                                               /*templateArgs=*/ArrayAttr{},
                                               /*operands=*/ValueRange{eventOperand});
          return;
        }
        auto args = rewriter.getArrayAttr({
            emitc::OpaqueAttr::get(ctx, pipeTok),
            eventLiteral,
        });
        rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "set_intra_block",
                                             /*args=*/args,
                                             /*templateArgs=*/ArrayAttr{},
                                             /*operands=*/ValueRange{});
      };

      if (eventIdAttr) {
        emitSet(Value{}, eventIdAttr, /*isDynamic=*/false);
        if (needsMirrorPlus16) {
          auto plus16 = IntegerAttr::get(eventIdAttr.getType(),
                                         eventIdAttr.getInt() + 16);
          emitSet(Value{}, plus16, /*isDynamic=*/false);
        }
      } else {
        Value eventI32 = castInterCoreEventIdToI32(rewriter, loc, eventIdDyn);
        emitSet(eventI32, IntegerAttr{}, /*isDynamic=*/true);
        if (needsMirrorPlus16) {
          auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
          Value c16 = makeEmitCIntConstant(rewriter, loc, i32Ty, 16);
          Value eventI32Plus16 =
              rewriter.create<emitc::AddOp>(loc, i32Ty, eventI32, c16).getResult();
          emitSet(eventI32Plus16, IntegerAttr{}, /*isDynamic=*/true);
        }
      }

      rewriter.eraseOp(op);
      return success();
    }

    InterCoreSyncCallDesc desc;
    if (eventIdAttr) {
      desc = buildInterCoreSyncSetCall(rewriter, loc, targetArch, op.getPipe(),
                                       eventIdAttr, fftsMode);
    } else {
      desc = buildInterCoreSyncSetCallDyn(rewriter, loc, targetArch, op.getPipe(),
                                          eventIdDyn, fftsMode);
    }
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, desc.callee,
                                         /*args=*/desc.args,
                                         /*templateArgs=*/ArrayAttr{},
                                         /*operands=*/desc.operands);

    rewriter.eraseOp(op);
    return success();
  }

  PTOArch targetArch;
};

struct PTOSyncWaitToEmitC : public OpConversionPattern<mlir::pto::SyncWaitOp> {
  PTOSyncWaitToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                     PTOArch targetArch)
      : OpConversionPattern<mlir::pto::SyncWaitOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult
  matchAndRewrite(mlir::pto::SyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    IntegerAttr eventIdAttr = op.getEventIdAttr();
    Value eventIdDyn = adaptor.getEventIdDyn();

    if ((eventIdAttr != nullptr) == static_cast<bool>(eventIdDyn))
      return rewriter.notifyMatchFailure(
          op, "expects exactly one of static event_id attr or dynamic event_id operand");

    InterCoreSyncCallDesc desc;
    if (eventIdAttr) {
      desc = buildInterCoreSyncWaitCall(rewriter, targetArch, op.getPipe(),
                                        eventIdAttr);
    } else {
      desc = buildInterCoreSyncWaitCallDyn(rewriter, loc, targetArch, op.getPipe(),
                                           eventIdDyn);
    }
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, desc.callee,
                                         desc.args, ArrayAttr{}, desc.operands);

    rewriter.eraseOp(op);
    return success();
  }

  PTOArch targetArch;
};

// GetBlockIdxOp Lowering (pto.get_block_idx -> get_block_idx())
struct PTOGetBlockIdxToEmitC
    : public OpConversionPattern<mlir::pto::GetBlockIdxOp> {
  using OpConversionPattern<mlir::pto::GetBlockIdxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetBlockIdxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_block_idx", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

// GetBlockNumOp Lowering (pto.get_block_num -> get_block_num())
struct PTOGetBlockNumToEmitC
    : public OpConversionPattern<mlir::pto::GetBlockNumOp> {
  using OpConversionPattern<mlir::pto::GetBlockNumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetBlockNumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_block_num", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

// GetSubBlockIdxOp Lowering (pto.get_block_idx -> get_subblockid())
struct PTOGetSubBlockIdxToEmitC
    : public OpConversionPattern<mlir::pto::GetSubBlockIdxOp> {
  using OpConversionPattern<mlir::pto::GetSubBlockIdxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetSubBlockIdxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_subblockid", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

// GetSubBlockNumOp Lowering.
struct PTOGetSubBlockNumToEmitC
    : public OpConversionPattern<mlir::pto::GetSubBlockNumOp> {
  using OpConversionPattern<mlir::pto::GetSubBlockNumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetSubBlockNumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_subblockdim", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};


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

    auto scatterAtomicTok = [&](pto::ScatterAtomicOp atomic) -> StringRef {
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
    };
    auto scatterOobTok = [&](pto::ScatterOOB mode) -> StringRef {
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
    };

    SmallVector<Attribute, 3> templateArgVec;
    const bool rowCoalesce =
        isRowCoalescedMGatherIndexType(op.getSrc().getType(), op.getIdx().getType());
    templateArgVec.push_back(emitc::OpaqueAttr::get(
        ctx, rowCoalesce ? "pto::Coalesce::Row" : "pto::Coalesce::Elem"));
    if (op.getScatterAtomicOp() != pto::ScatterAtomicOp::None ||
        op.getScatterOob() != pto::ScatterOOB::Undefined) {
      templateArgVec.push_back(emitc::OpaqueAttr::get(
          ctx, scatterAtomicTok(op.getScatterAtomicOp())));
      if (op.getScatterOob() != pto::ScatterOOB::Undefined)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, scatterOobTok(op.getScatterOob())));
    }
    ArrayAttr templateArgs = rewriter.getArrayAttr(templateArgVec);

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "MSCATTER",
        ArrayAttr{}, templateArgs,
        ValueRange{memArg, src, idx});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOSetValToSETVAL : public OpConversionPattern<pto::TSetValOp> {
  using OpConversionPattern<pto::TSetValOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSetValOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value dst = peelUnrealized(adaptor.getDst());
    Value val = peelUnrealized(adaptor.getVal());

    // ---- offset: SSA index operand ----
    Value offset = peelUnrealized(adaptor.getOffset());

    // Emit a marker call and let the ptoas post-processing step lower it to
    // the corresponding tile setter.
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__TILE_SET_VALUE",
        ArrayAttr{}, ArrayAttr{}, ValueRange{dst, offset, val});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOGetValToGETVAL : public OpConversionPattern<pto::TGetValOp> {
  using OpConversionPattern<pto::TGetValOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGetValOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());

    // ---- offset: SSA index operand ----
    Value offset = peelUnrealized(adaptor.getOffset());

    // Emit a marker call and let the ptoas post-processing step lower it to
    // the corresponding tile getter.
    Type dstTy = getTypeConverter()->convertType(op.getDst().getType());
    if (!dstTy)
      return failure();
    auto call = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(),
        TypeRange{dstTy},
        "PTOAS__TILE_GET_VALUE",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{src, offset});

    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct PTOTAxpyToEmitC : public OpConversionPattern<pto::TAxpyOp> {
  using OpConversionPattern<pto::TAxpyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAxpyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TAXPY",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOHistogramToEmitC : public OpConversionPattern<pto::THistogramOp> {
  using OpConversionPattern<pto::THistogramOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::THistogramOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value idx = peelUnrealized(adaptor.getIdx());
    Value dst = peelUnrealized(adaptor.getDst());

    StringRef histByte = "HistByte::BYTE_1";
    int64_t byte = 1;
    auto byteAttr = op.getByteAttr();
    if (byteAttr)
      byte = byteAttr.getInt();
    if (auto legacyIsMSB = op->getAttrOfType<BoolAttr>("isMSB")) {
      int64_t legacyByte = legacyIsMSB.getValue() ? 1 : 0;
      if (byteAttr && byte != legacyByte)
        return rewriter.notifyMatchFailure(
            op, "conflicting 'byte' and legacy 'isMSB' attributes");
      byte = legacyByte;
    }
    switch (byte) {
    case 0:
      histByte = "HistByte::BYTE_0";
      break;
    case 1:
      histByte = "HistByte::BYTE_1";
      break;
    case 2:
      histByte = "HistByte::BYTE_2";
      break;
    case 3:
      histByte = "HistByte::BYTE_3";
      break;
    default:
      return rewriter.notifyMatchFailure(op, "expected byte to be in range [0, 3]");
    }

    auto templateArgs =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, histByte)});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "THISTOGRAM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/ValueRange{dst, src, idx});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOGetScaleAddrToEmitC
    : public OpConversionPattern<pto::TGetScaleAddrOp> {
  using OpConversionPattern<pto::TGetScaleAddrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGetScaleAddrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGET_SCALE_ADDR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSetValidShapeToEmitC : public OpConversionPattern<pto::SetValidShapeOp> {
  using OpConversionPattern<pto::SetValidShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SetValidShapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto peelAllCasts = [](Value v) {
      while (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
        v = castOp.getOperand(0);
      if (auto castOp = v.getDefiningOp<emitc::CastOp>())
        v = castOp.getOperand();
      return v;
    };
    auto isTileLike = [](Value v) -> bool {
      auto ot = dyn_cast<emitc::OpaqueType>(v.getType());
      if (!ot)
        return false;
      StringRef s = ot.getValue();
      return s.contains("Tile<") || s.contains("ConvTile<");
    };

    Value src = peelAllCasts(peelUnrealized(adaptor.getSource()));
    Value row = peelUnrealized(adaptor.getValidRow());
    Value col = peelUnrealized(adaptor.getValidCol());

    if (!isTileLike(src))
      return rewriter.notifyMatchFailure(
          op, "set_validshape source must lower to a tile-like value");

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__TILE_SET_VALIDSHAPE", ArrayAttr{},
        ArrayAttr{}, ValueRange{src, row, col});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOGetValidShapeToEmitC
    : public OpConversionPattern<pto::GetValidShapeOp> {
  using OpConversionPattern<pto::GetValidShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::GetValidShapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto peelAllCasts = [](Value v) {
      while (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
        v = castOp.getOperand(0);
      if (auto castOp = v.getDefiningOp<emitc::CastOp>())
        v = castOp.getOperand();
      return v;
    };
    auto isTileLike = [](Value v) -> bool {
      auto ot = dyn_cast<emitc::OpaqueType>(v.getType());
      if (!ot)
        return false;
      StringRef s = ot.getValue();
      return s.contains("Tile<") || s.contains("ConvTile<");
    };

    Value src = peelAllCasts(peelUnrealized(adaptor.getSource()));
    if (!isTileLike(src))
      return rewriter.notifyMatchFailure(
          op, "get_validshape source must lower to a tile-like value");

    auto resultTy = getTypeConverter()->convertType(rewriter.getIndexType());
    if (!resultTy)
      return failure();

    Value row = rewriter
                    .create<emitc::CallOpaqueOp>(
                        op.getLoc(), resultTy,
                        "PTOAS__TILE_GET_VALID_ROW", ArrayAttr{},
                        ArrayAttr{}, ValueRange{src})
                    .getResult(0);
    Value col = rewriter
                    .create<emitc::CallOpaqueOp>(
                        op.getLoc(), resultTy,
                        "PTOAS__TILE_GET_VALID_COL", ArrayAttr{},
                        ArrayAttr{}, ValueRange{src})
                    .getResult(0);
    rewriter.replaceOp(op, ValueRange{row, col});
    return success();
  }
};

struct PTOTAssignToEmitC : public OpConversionPattern<pto::TAssignOp> {
  using OpConversionPattern<pto::TAssignOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAssignOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto peelAllCasts = [](Value v) {
      while (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
        v = castOp.getOperand(0);
      if (auto castOp = v.getDefiningOp<emitc::CastOp>())
        v = castOp.getOperand();
      return v;
    };
    auto isTileLike = [](Value v) -> bool {
      auto ot = dyn_cast<emitc::OpaqueType>(v.getType());
      if (!ot)
        return false;
      StringRef s = ot.getValue();
      return s.contains("Tile<") || s.contains("ConvTile<");
    };

    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value tile = peelAllCasts(peelUnrealized(adaptor.getTile()));
    if (!isTileLike(tile))
      return rewriter.notifyMatchFailure(
          op, "tassign tile must lower to a tile-like value");

    Value addr = peelUnrealized(adaptor.getAddr());
    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
    if (isa<emitc::PointerType>(addr.getType()) ||
        (isa<emitc::OpaqueType>(addr.getType()) &&
         cast<emitc::OpaqueType>(addr.getType()).getValue().ends_with("*"))) {
      auto rcU64 =
          rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
      addr = rewriter
                 .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                              ArrayAttr{}, rcU64,
                                              ValueRange{addr})
                 .getResult(0);
    } else if (addr.getType() != u64Ty) {
      addr = rewriter.create<emitc::CastOp>(loc, u64Ty, addr).getResult();
    }

    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{tile, addr});
    rewriter.replaceOp(op, tile);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.load_scalar / pto.store_scalar lowering -> ptr[offset]
//===----------------------------------------------------------------------===//

static Type getPointerLikeElementType(Type type) {
  if (auto ptrTy = dyn_cast<pto::PtrType>(type))
    return ptrTy.getElementType();
  if (auto memTy = dyn_cast<MemRefType>(type))
    return memTy.getElementType();
  return Type();
}

struct PTOPtrToIntToEmitC : public OpConversionPattern<pto::PtrToIntOp> {
  using OpConversionPattern<pto::PtrToIntOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PtrToIntOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = peelUnrealized(adaptor.getPtr());
    Type dstTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!dstTy)
      return failure();

    auto dstOpaque = dyn_cast<emitc::OpaqueType>(dstTy);
    if (!dstOpaque)
      return failure();

    auto templateArgs =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                      dstOpaque.getValue())});
    auto cast = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), dstTy, "reinterpret_cast", ArrayAttr{}, templateArgs,
        ValueRange{ptr});
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

struct PTOIntToPtrToEmitC : public OpConversionPattern<pto::IntToPtrOp> {
  using OpConversionPattern<pto::IntToPtrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::IntToPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value addr = peelUnrealized(adaptor.getAddr());
    Type dstTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!dstTy)
      return failure();

    Type dstElemTy = getPointerLikeElementType(op.getResult().getType());
    if (!dstElemTy)
      return failure();

    std::string castType =
        std::string("__gm__ ") + getEmitCScalarTypeToken(dstElemTy) + "*";
    auto templateArgs =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                      castType)});
    auto cast = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), dstTy, "reinterpret_cast", ArrayAttr{}, templateArgs,
        ValueRange{addr});
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

struct PTOLoadScalarToEmitC : public OpConversionPattern<pto::LoadScalarOp> {
  using OpConversionPattern<pto::LoadScalarOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::LoadScalarOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = peelUnrealized(adaptor.getPtr());
    Value offset = peelUnrealized(adaptor.getOffset());

    Type dstTy = getTypeConverter()->convertType(op.getValue().getType());
    if (!dstTy)
      return failure();

    auto call = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{dstTy}, "PTOAS__PTR_LOAD",
        ArrayAttr{}, ArrayAttr{}, ValueRange{ptr, offset});

    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct PTOStoreScalarToEmitC : public OpConversionPattern<pto::StoreScalarOp> {
  using OpConversionPattern<pto::StoreScalarOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::StoreScalarOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = peelUnrealized(adaptor.getPtr());
    Value offset = peelUnrealized(adaptor.getOffset());
    Value val = peelUnrealized(adaptor.getValue());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__PTR_STORE",
        ArrayAttr{}, ArrayAttr{}, ValueRange{ptr, offset, val});
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__SCALAR_GM_STORE_FLUSH",
        ArrayAttr{}, ArrayAttr{}, ValueRange{ptr});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.tabs lowering -> TABS(dst, src)
//===----------------------------------------------------------------------===//

struct PTOTAbsToTABS : public OpConversionPattern<pto::TAbsOp> {
  using OpConversionPattern<pto::TAbsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAbsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // intrinsic: TABS(dst, src)
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TABS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tadd lowering -> TADD(dst, src0, src1)
//===----------------------------------------------------------------------===//

struct PTOTAddToTADD : public OpConversionPattern<pto::TAddOp> {
  using OpConversionPattern<pto::TAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TADD", ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOInitializeL2G2LPipeToEmitC
    : public OpConversionPattern<mlir::pto::InitializeL2G2LPipeOp> {
  PTOInitializeL2G2LPipeToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                PTOArch targetArch)
      : OpConversionPattern<mlir::pto::InitializeL2G2LPipeOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::InitializeL2G2LPipeOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tpipeTok = buildTPipeTokenFromInitOp(op.getOperation(), targetArch);
    if (failed(tpipeTok))
      return rewriter.notifyMatchFailure(op, "failed to build TPipe token");

    auto *ctx = rewriter.getContext();
    auto emitPipeTy =
        cast<Type>(getTypeConverter()->convertType(op.getPipe().getType()));

    Value gmAddr = peelUnrealized(adaptor.getGmAddr());
    gmAddr = materializeTensorViewDataPointer(
        rewriter, op.getLoc(), gmAddr, op.getGmAddr().getType());
    Value localAddr =
        op.getLocalAddr() ? peelUnrealized(adaptor.getLocalAddr()) : Value();
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value zero = makeEmitCIntConstant(rewriter, op.getLoc(), i32Ty, 0);

    Value c2vBuf = zero;
    Value v2cBuf = zero;
    if (op.getDirMask() == 1)
      c2vBuf = localAddr ? localAddr : zero;
    else if (op.getDirMask() == 2)
      v2cBuf = localAddr ? localAddr : zero;
    else if (op.getDirMask() == 3) {
      if (localAddr) {
        if (!op.getPeerLocalAddr())
          return rewriter.notifyMatchFailure(
              op, "bidirectional l2g2l pipe requires peer local buffer");
        c2vBuf = localAddr;
        v2cBuf = peelUnrealized(adaptor.getPeerLocalAddr());
      }
    } else
      return rewriter.notifyMatchFailure(op, "unsupported dir_mask");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{emitPipeTy}, *tpipeTok, ArrayAttr{}, ArrayAttr{},
        ValueRange{gmAddr, c2vBuf, v2cBuf});
    return success();
  }

  PTOArch targetArch;
};

struct PTOInitializeL2LPipeToEmitC
    : public OpConversionPattern<mlir::pto::InitializeL2LPipeOp> {
  PTOInitializeL2LPipeToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                              PTOArch targetArch)
      : OpConversionPattern<mlir::pto::InitializeL2LPipeOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::InitializeL2LPipeOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tpipeTok = buildTPipeTokenFromInitOp(op.getOperation(), targetArch);
    if (failed(tpipeTok))
      return rewriter.notifyMatchFailure(op, "failed to build TPipe token");

    auto *ctx = rewriter.getContext();
    auto emitPipeTy =
        cast<Type>(getTypeConverter()->convertType(op.getPipe().getType()));

    auto gmPtrTy =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "__gm__ void"));
    Value nullGm =
        makeEmitCOpaqueConstant(rewriter, op.getLoc(), gmPtrTy, "nullptr");
    auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
    Value zero = makeEmitCIntConstant(rewriter, op.getLoc(), i32Ty, 0);
    Value localAddr = peelUnrealized(adaptor.getLocalAddr());

    Value c2vBuf = zero;
    Value v2cBuf = zero;
    if (op.getDirMask() == 1)
      c2vBuf = localAddr;
    else if (op.getDirMask() == 2)
      v2cBuf = localAddr;
    else if (op.getDirMask() == 3) {
      c2vBuf = localAddr;
      v2cBuf = peelUnrealized(adaptor.getPeerLocalAddr());
    } else
      return rewriter.notifyMatchFailure(op, "unsupported dir_mask");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{emitPipeTy}, *tpipeTok, ArrayAttr{}, ArrayAttr{},
        ValueRange{nullGm, c2vBuf, v2cBuf});
    return success();
  }

  PTOArch targetArch;
};

struct PTOBuildAsyncSessionToEmitC
    : public OpConversionPattern<mlir::pto::BuildAsyncSessionOp> {
  PTOBuildAsyncSessionToEmitC(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<mlir::pto::BuildAsyncSessionOp>(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(mlir::pto::BuildAsyncSessionOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Location loc = op.getLoc();

    auto sessionTy =
        dyn_cast<emitc::OpaqueType>(getTypeConverter()->convertType(op.getSession().getType()));
    if (!sessionTy)
      return rewriter.notifyMatchFailure(op, "failed to convert async session type");

    FailureOr<Value> scratchTile =
        buildAsyncScratchTileValue(rewriter, loc, op.getScratch(),
                                   adaptor.getScratch());
    if (failed(scratchTile))
      return rewriter.notifyMatchFailure(op, "failed to materialize async scratch tile");

    Value workspace =
        castToGMBytePointer(rewriter, loc, peelUnrealized(adaptor.getWorkspace()));

    Value session = rewriter
                        .create<emitc::VariableOp>(
                            loc, sessionTy, emitc::OpaqueAttr::get(ctx, ""))
                        .getResult();

    auto u32Ty = emitc::OpaqueType::get(ctx, "uint32_t");

    auto makeU32Const = [&](uint64_t value) -> Value {
      return makeEmitCOpaqueConstant(rewriter, loc, u32Ty,
                                     std::to_string(value) + "u");
    };
    uint64_t syncId = op.getSyncIdAttr() ? op.getSyncIdAttr().getInt() : 0;
    uint64_t blockBytes =
        op.getBlockBytesAttr() ? op.getBlockBytesAttr().getInt() : 32 * 1024;
    uint64_t commBlockOffset =
        op.getCommBlockOffsetAttr() ? op.getCommBlockOffsetAttr().getInt() : 0;
    uint64_t queueNum = op.getQueueNumAttr() ? op.getQueueNumAttr().getInt() : 1;
    uint64_t channelGroupIdx = op.getChannelGroupIdxAttr()
                                   ? op.getChannelGroupIdxAttr().getInt()
                                   : UINT32_MAX;

    Value syncIdVal = makeU32Const(syncId);
    Value channelGroupIdxVal =
        channelGroupIdx == UINT32_MAX
            ? makeEmitCOpaqueConstant(rewriter, loc, u32Ty, "UINT32_MAX")
            : makeU32Const(channelGroupIdx);

    auto baseConfigTy =
        emitc::OpaqueType::get(ctx, "pto::comm::sdma::SdmaBaseConfig");
    Value baseConfig =
        rewriter
            .create<emitc::VariableOp>(
                loc, baseConfigTy,
                emitc::OpaqueAttr::get(
                    ctx, "{" + std::to_string(blockBytes) + "ULL, " +
                             std::to_string(commBlockOffset) + "ULL, " +
                             std::to_string(queueNum) + "u}"))
            .getResult();

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "pto::comm::BuildAsyncSession<pto::comm::DmaEngine::SDMA>",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{*scratchTile, workspace, session, syncIdVal, baseConfig,
                   channelGroupIdxVal});

    rewriter.replaceOp(op, session);
    return success();
  }
};

template <typename AsyncOp>
struct PTOAsyncTransferToEmitC : public OpConversionPattern<AsyncOp> {
  using OpConversionPattern<AsyncOp>::OpConversionPattern;

  explicit PTOAsyncTransferToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                   StringRef callee)
      : OpConversionPattern<AsyncOp>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(AsyncOp op, typename AsyncOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());
    Value dstGT = dst;
    Value srcGT = src;
    if (!isEmitCGlobalTensorLikeType(dstGT.getType())) {
      auto dstMrTy = dyn_cast<MemRefType>(op.getDst().getType());
      if (!dstMrTy)
        return rewriter.notifyMatchFailure(op, "expected dst to lower to GlobalTensor or memref");
      dstGT = buildGlobalTensorFromMemref(rewriter, op.getLoc(), dst, dstMrTy,
                                          op.getDst().getDefiningOp()
                                              ? op.getDst().getDefiningOp()
                                              : op.getOperation());
    }
    if (!isEmitCGlobalTensorLikeType(srcGT.getType())) {
      auto srcMrTy = dyn_cast<MemRefType>(op.getSrc().getType());
      if (!srcMrTy)
        return rewriter.notifyMatchFailure(op, "expected src to lower to GlobalTensor or memref");
      srcGT = buildGlobalTensorFromMemref(rewriter, op.getLoc(), src, srcMrTy,
                                          op.getSrc().getDefiningOp()
                                              ? op.getSrc().getDefiningOp()
                                              : op.getOperation());
    }
    if (!dstGT || !srcGT)
      return rewriter.notifyMatchFailure(op, "failed to build GlobalTensor operands");

    Type eventTy = this->getTypeConverter()->convertType(op.getEvent().getType());
    if (!eventTy)
      return rewriter.notifyMatchFailure(op, "failed to convert async event type");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{eventTy}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{dstGT, srcGT, peelUnrealized(adaptor.getSession())});
    return success();
  }

  std::string callee;
};

template <typename AsyncEventOp>
struct PTOAsyncEventToEmitC : public OpConversionPattern<AsyncEventOp> {
  explicit PTOAsyncEventToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                StringRef callee)
      : OpConversionPattern<AsyncEventOp>(typeConverter, ctx),
        callee(callee.str()) {}

  LogicalResult matchAndRewrite(AsyncEventOp op,
                                typename AsyncEventOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultTy =
        this->getTypeConverter()->convertType(op.getCompleted().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "failed to convert async event result type");

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{resultTy}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{peelUnrealized(adaptor.getEvent()),
                   peelUnrealized(adaptor.getSession())});
    return success();
  }

  std::string callee;
};

static FailureOr<Value> buildCommGlobalTensorValue(
    ConversionPatternRewriter &rewriter, Location loc, Value originalValue,
    Value emittedValue, Operation *anchor) {
  Value value = peelUnrealized(emittedValue);
  if (isEmitCGlobalTensorLikeType(value.getType()))
    return value;

  auto memTy = dyn_cast<MemRefType>(originalValue.getType());
  if (!memTy)
    return failure();

  Value gt = buildGlobalTensorFromMemref(rewriter, loc, value, memTy, anchor);
  if (!gt)
    return failure();
  return gt;
}

static FailureOr<Value> buildCommTileValue(ConversionPatternRewriter &rewriter,
                                           Location loc, Value originalValue,
                                           Value emittedValue) {
  Value value = peelUnrealized(emittedValue);
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(value.getType())) {
    StringRef typeStr = opaqueTy.getValue();
    if (typeStr.contains("Tile<") || typeStr.contains("ConvTile<"))
      return value;
  }
  return buildAsyncScratchTileValue(rewriter, loc, originalValue, emittedValue);
}

static FailureOr<Value> buildCollectiveParallelGroup(
    ConversionPatternRewriter &rewriter, Location loc,
    ArrayRef<Value> groupGTs, int64_t root) {
  if (groupGTs.empty())
    return failure();

  auto firstTy = dyn_cast<emitc::OpaqueType>(groupGTs.front().getType());
  if (!firstTy)
    return failure();

  auto *ctx = rewriter.getContext();
  auto arrayTy = emitc::ArrayType::get({static_cast<int64_t>(groupGTs.size())},
                                       firstTy);
  auto groupArray = cast<TypedValue<emitc::ArrayType>>(
      rewriter
          .create<emitc::VariableOp>(loc, arrayTy,
                                     emitc::OpaqueAttr::get(ctx, "{}"))
          .getResult());

  auto indexTy = emitc::OpaqueType::get(ctx, "int");
  for (auto [idx, groupVal] : llvm::enumerate(groupGTs)) {
    Value idxVal =
        makeEmitCIntConstant(rewriter, loc, indexTy, static_cast<int64_t>(idx));
    Value slot =
        rewriter.create<emitc::SubscriptOp>(loc, groupArray, ValueRange{idxVal})
            .getResult();
    rewriter.create<emitc::AssignOp>(loc, slot, groupVal);
  }

  std::string pgTypeStr =
      (Twine("pto::comm::ParallelGroup<") + firstTy.getValue() + ">").str();
  auto pgTy = emitc::OpaqueType::get(ctx, pgTypeStr);
  Value sizeVal = makeEmitCIntConstant(rewriter, loc, indexTy,
                                       static_cast<int64_t>(groupGTs.size()));
  Value rootVal = makeEmitCIntConstant(rewriter, loc, indexTy, root);
  return rewriter
      .create<emitc::CallOpaqueOp>(
          loc, TypeRange{pgTy}, (Twine(pgTypeStr) + "::Create").str(),
          ArrayAttr{}, ArrayAttr{}, ValueRange{groupArray, sizeVal, rootVal})
      .getResult(0);
}

static std::string notifyOpTok(pto::NotifyOp op) {
  switch (op) {
  case pto::NotifyOp::AtomicAdd:
    return "pto::comm::NotifyOp::AtomicAdd";
  case pto::NotifyOp::Set:
    return "pto::comm::NotifyOp::Set";
  }
  return "pto::comm::NotifyOp::Set";
}

static void emitPipeBarrier(ConversionPatternRewriter &rewriter, Location loc,
                            StringRef pipeTok) {
  auto *ctx = rewriter.getContext();
  auto args = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, pipeTok)});
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "pipe_barrier", args,
                                       ArrayAttr{}, ValueRange{});
}

// Issue #711: TNOTIFY writes its signal on the scalar pipe, and
// TNOTIFY_IMPL's trailing pipe_barrier(PIPE_ALL) runs *after* that store.
// If prior pto.tload / pto.tstore work is still in flight on an MTE pipe when
// the signal lands, the receiver's matching TWAIT can return before the data
// is visible. Emit only the MTE pipe drains that the pre-lowering analysis
// proved may be needed before this TNotify.
static void emitTNotifyMteDrain(ConversionPatternRewriter &rewriter,
                                Location loc, unsigned mask) {
  if (mask & kDrainMte2)
    emitPipeBarrier(rewriter, loc, "PIPE_MTE2");
  if (mask & kDrainMte3)
    emitPipeBarrier(rewriter, loc, "PIPE_MTE3");
}

static std::string waitCmpTok(pto::WaitCmp cmp) {
  switch (cmp) {
  case pto::WaitCmp::EQ:
    return "pto::comm::WaitCmp::EQ";
  case pto::WaitCmp::NE:
    return "pto::comm::WaitCmp::NE";
  case pto::WaitCmp::GT:
    return "pto::comm::WaitCmp::GT";
  case pto::WaitCmp::GE:
    return "pto::comm::WaitCmp::GE";
  case pto::WaitCmp::LT:
    return "pto::comm::WaitCmp::LT";
  case pto::WaitCmp::LE:
    return "pto::comm::WaitCmp::LE";
  }
  return "pto::comm::WaitCmp::EQ";
}

static std::string reduceOpTok(pto::ReduceOp op) {
  switch (op) {
  case pto::ReduceOp::Sum:
    return "pto::comm::ReduceOp::Sum";
  case pto::ReduceOp::Max:
    return "pto::comm::ReduceOp::Max";
  case pto::ReduceOp::Min:
    return "pto::comm::ReduceOp::Min";
  }
  return "pto::comm::ReduceOp::Sum";
}

template <typename OpTy>
static FailureOr<SmallVector<Value>> buildCommGroupGlobalTensors(
    ConversionPatternRewriter &rewriter, Location loc, OpTy op,
    ValueRange originalGroup, ValueRange emittedGroup) {
  SmallVector<Value> groupGTs;
  groupGTs.reserve(originalGroup.size());
  for (auto [orig, emitted] : llvm::zip(originalGroup, emittedGroup)) {
    FailureOr<Value> gt =
        buildCommGlobalTensorValue(rewriter, loc, orig, emitted, op.getOperation());
    if (failed(gt))
      return failure();
    groupGTs.push_back(*gt);
  }
  return groupGTs;
}

template <typename CollectiveOp>
struct PTOCommCollectiveToEmitC : public OpConversionPattern<CollectiveOp> {
  using OpConversionPattern<CollectiveOp>::OpConversionPattern;

  explicit PTOCommCollectiveToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                    StringRef apiName)
      : OpConversionPattern<CollectiveOp>(typeConverter, ctx),
        apiName(apiName.str()) {}

  LogicalResult matchAndRewrite(CollectiveOp op, typename CollectiveOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto buildPong = [&](Value original, Value emitted, StringRef name) -> FailureOr<Value> {
      if (!original)
        return failure();
      return buildCommTileValue(rewriter, loc, original, emitted);
    };

    if constexpr (std::is_same_v<CollectiveOp, pto::TBroadcastOp>) {
      FailureOr<Value> srcGT =
          buildCommGlobalTensorValue(rewriter, loc, op.getSrc(), adaptor.getSrc(),
                                     op.getOperation());
      FailureOr<Value> pingTile =
          buildCommTileValue(rewriter, loc, op.getPing(), adaptor.getPing());
      auto groupGTs =
          buildCommGroupGlobalTensors(rewriter, loc, op, op.getGroup(), adaptor.getGroup());
      if (failed(srcGT) || failed(pingTile) || failed(groupGTs))
        return rewriter.notifyMatchFailure(op, "failed to materialize broadcast operands");
      FailureOr<Value> pg = buildCollectiveParallelGroup(rewriter, loc, *groupGTs, op.getRoot());
      if (failed(pg))
        return rewriter.notifyMatchFailure(op, "failed to materialize broadcast group");
      if (op.getPong()) {
        FailureOr<Value> pongTile =
            buildPong(op.getPong(), adaptor.getPong(), "__pong");
        if (failed(pongTile))
          return rewriter.notifyMatchFailure(op, "failed to materialize pong tile");
        rewriter.create<emitc::CallOpaqueOp>(
            loc, TypeRange{}, "pto::comm::TBROADCAST", ArrayAttr{}, ArrayAttr{},
            ValueRange{*pg, *srcGT, *pingTile, *pongTile});
      } else {
        rewriter.create<emitc::CallOpaqueOp>(
            loc, TypeRange{}, "pto::comm::TBROADCAST", ArrayAttr{}, ArrayAttr{},
            ValueRange{*pg, *srcGT, *pingTile});
      }
    } else if constexpr (std::is_same_v<CollectiveOp, pto::CommTGatherOp>) {
      FailureOr<Value> dstGT =
          buildCommGlobalTensorValue(rewriter, loc, op.getDst(), adaptor.getDst(),
                                     op.getOperation());
      FailureOr<Value> pingTile =
          buildCommTileValue(rewriter, loc, op.getPing(), adaptor.getPing());
      auto groupGTs =
          buildCommGroupGlobalTensors(rewriter, loc, op, op.getGroup(), adaptor.getGroup());
      if (failed(dstGT) || failed(pingTile) || failed(groupGTs))
        return rewriter.notifyMatchFailure(op, "failed to materialize gather operands");
      FailureOr<Value> pg = buildCollectiveParallelGroup(rewriter, loc, *groupGTs, op.getRoot());
      if (failed(pg))
        return rewriter.notifyMatchFailure(op, "failed to materialize gather group");
      if (op.getPong()) {
        FailureOr<Value> pongTile =
            buildPong(op.getPong(), adaptor.getPong(), "__pong");
        if (failed(pongTile))
          return rewriter.notifyMatchFailure(op, "failed to materialize pong tile");
        rewriter.create<emitc::CallOpaqueOp>(
            loc, TypeRange{}, "pto::comm::TGATHER", ArrayAttr{}, ArrayAttr{},
            ValueRange{*pg, *dstGT, *pingTile, *pongTile});
      } else {
        rewriter.create<emitc::CallOpaqueOp>(
            loc, TypeRange{}, "pto::comm::TGATHER", ArrayAttr{}, ArrayAttr{},
            ValueRange{*pg, *dstGT, *pingTile});
      }
    } else if constexpr (std::is_same_v<CollectiveOp, pto::CommTScatterOp>) {
      FailureOr<Value> srcGT =
          buildCommGlobalTensorValue(rewriter, loc, op.getSrc(), adaptor.getSrc(),
                                     op.getOperation());
      FailureOr<Value> pingTile =
          buildCommTileValue(rewriter, loc, op.getPing(), adaptor.getPing());
      auto groupGTs =
          buildCommGroupGlobalTensors(rewriter, loc, op, op.getGroup(), adaptor.getGroup());
      if (failed(srcGT) || failed(pingTile) || failed(groupGTs))
        return rewriter.notifyMatchFailure(op, "failed to materialize scatter operands");
      FailureOr<Value> pg = buildCollectiveParallelGroup(rewriter, loc, *groupGTs, op.getRoot());
      if (failed(pg))
        return rewriter.notifyMatchFailure(op, "failed to materialize scatter group");
      if (op.getPong()) {
        FailureOr<Value> pongTile =
            buildPong(op.getPong(), adaptor.getPong(), "__pong");
        if (failed(pongTile))
          return rewriter.notifyMatchFailure(op, "failed to materialize pong tile");
        rewriter.create<emitc::CallOpaqueOp>(
            loc, TypeRange{}, "pto::comm::TSCATTER", ArrayAttr{}, ArrayAttr{},
            ValueRange{*pg, *srcGT, *pingTile, *pongTile});
      } else {
        rewriter.create<emitc::CallOpaqueOp>(
            loc, TypeRange{}, "pto::comm::TSCATTER", ArrayAttr{}, ArrayAttr{},
            ValueRange{*pg, *srcGT, *pingTile});
      }
    } else {
      FailureOr<Value> dstGT =
          buildCommGlobalTensorValue(rewriter, loc, op.getDst(), adaptor.getDst(),
                                     op.getOperation());
      FailureOr<Value> accTile =
          buildCommTileValue(rewriter, loc, op.getAcc(), adaptor.getAcc());
      FailureOr<Value> recvPing =
          buildCommTileValue(rewriter, loc, op.getRecvPing(), adaptor.getRecvPing());
      auto groupGTs =
          buildCommGroupGlobalTensors(rewriter, loc, op, op.getGroup(), adaptor.getGroup());
      if (failed(dstGT) || failed(accTile) || failed(recvPing) || failed(groupGTs))
        return rewriter.notifyMatchFailure(op, "failed to materialize reduce operands");
      FailureOr<Value> pg = buildCollectiveParallelGroup(rewriter, loc, *groupGTs, op.getRoot());
      if (failed(pg))
        return rewriter.notifyMatchFailure(op, "failed to materialize reduce group");
      if (op.getRecvPong()) {
        FailureOr<Value> recvPong =
            buildPong(op.getRecvPong(), adaptor.getRecvPong(), "__recv_pong");
        if (failed(recvPong))
          return rewriter.notifyMatchFailure(op, "failed to materialize recv_pong");
        auto reduceTy =
            emitc::OpaqueType::get(rewriter.getContext(), "pto::comm::ReduceOp");
        Value reduceOp = makeEmitCOpaqueConstant(rewriter, loc, reduceTy,
                                                reduceOpTok(op.getReduceOp()));
        rewriter.create<emitc::CallOpaqueOp>(
            loc, TypeRange{}, "pto::comm::TREDUCE", ArrayAttr{}, ArrayAttr{},
            ValueRange{*pg, *dstGT, *accTile, *recvPing, *recvPong, reduceOp});
      } else {
        auto reduceTy =
            emitc::OpaqueType::get(rewriter.getContext(), "pto::comm::ReduceOp");
        Value reduceOp = makeEmitCOpaqueConstant(rewriter, loc, reduceTy,
                                                reduceOpTok(op.getReduceOp()));
        rewriter.create<emitc::CallOpaqueOp>(
            loc, TypeRange{}, "pto::comm::TREDUCE", ArrayAttr{}, ArrayAttr{},
            ValueRange{*pg, *dstGT, *accTile, *recvPing, reduceOp});
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  std::string apiName;
};

template <typename OpTy>
struct PTOP2PCommToEmitC : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  explicit PTOP2PCommToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                             StringRef callee)
      : OpConversionPattern<OpTy>(typeConverter, ctx), callee(callee.str()) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> dstGT =
        buildCommGlobalTensorValue(rewriter, op.getLoc(), op.getDst(), adaptor.getDst(),
                                   op.getOperation());
    FailureOr<Value> srcGT =
        buildCommGlobalTensorValue(rewriter, op.getLoc(), op.getSrc(), adaptor.getSrc(),
                                   op.getOperation());
    FailureOr<Value> pingTile =
        buildCommTileValue(rewriter, op.getLoc(), op.getPing(), adaptor.getPing());
    if (failed(dstGT) || failed(srcGT) || failed(pingTile))
      return rewriter.notifyMatchFailure(op, "failed to materialize p2p operands");

    SmallVector<Value> operands{*dstGT, *srcGT, *pingTile};
    std::string actualCallee = callee;
    if constexpr (std::is_same_v<OpTy, pto::TPutOp>) {
      if (op.getAtomicType() == pto::AtomicType::AtomicAdd)
        actualCallee = "pto::comm::TPUT<pto::AtomicType::AtomicAdd>";
    }
    if (op.getPong()) {
      FailureOr<Value> pongTile =
          buildCommTileValue(rewriter, op.getLoc(), op.getPong(), adaptor.getPong());
      if (failed(pongTile))
        return rewriter.notifyMatchFailure(op, "failed to materialize pong tile");
      operands.push_back(*pongTile);
    }

    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, actualCallee,
                                         ArrayAttr{}, ArrayAttr{}, operands);
    rewriter.eraseOp(op);
    return success();
  }

  std::string callee;
};

template <typename SignalOp>
struct PTOSignalCommToEmitC : public OpConversionPattern<SignalOp> {
  using OpConversionPattern<SignalOp>::OpConversionPattern;

  explicit PTOSignalCommToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                                StringRef callee)
      : OpConversionPattern<SignalOp>(typeConverter, ctx),
        callee(callee.str()) {}

  LogicalResult matchAndRewrite(SignalOp op, typename SignalOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    FailureOr<Value> signalGT = buildCommGlobalTensorValue(
        rewriter, op.getLoc(), op.getSignal(), adaptor.getSignal(), op.getOperation());
    if (failed(signalGT))
      return rewriter.notifyMatchFailure(op, "failed to materialize signal operand");

    if constexpr (std::is_same_v<SignalOp, pto::TNotifyOp>) {
      auto notifyTy =
          emitc::OpaqueType::get(rewriter.getContext(), "pto::comm::NotifyOp");
      Value notifyOp = makeEmitCOpaqueConstant(
          rewriter, op.getLoc(), notifyTy, notifyOpTok(op.getNotifyOp()));
      SmallVector<Value> operands{*signalGT, peelUnrealized(adaptor.getValue()),
                                  notifyOp};
      // See emitTNotifyMteDrain comment: drain in-flight MTE work before the
      // scalar-pipe signal store so the notify/wait handshake is honored.
      unsigned drainMask = 0;
      if (op->hasAttr(kTNotifyDrainMte2AttrName))
        drainMask |= kDrainMte2;
      if (op->hasAttr(kTNotifyDrainMte3AttrName))
        drainMask |= kDrainMte3;
      emitTNotifyMteDrain(rewriter, op.getLoc(), drainMask);
      rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                           ArrayAttr{}, ArrayAttr{}, operands);
      rewriter.eraseOp(op);
    } else {
      auto waitCmpTy =
          emitc::OpaqueType::get(rewriter.getContext(), "pto::comm::WaitCmp");
      Value waitCmp = makeEmitCOpaqueConstant(
          rewriter, op.getLoc(), waitCmpTy, waitCmpTok(op.getCmp()));
      SmallVector<Value> operands{*signalGT, peelUnrealized(adaptor.getCmpValue()),
                                  waitCmp};
      if constexpr (std::is_same_v<SignalOp, pto::TTestOp>) {
        Type resultTy = this->getTypeConverter()->convertType(op.getResult().getType());
        if (!resultTy)
          return rewriter.notifyMatchFailure(op, "failed to convert ttest result type");
        rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
            op, TypeRange{resultTy}, callee, ArrayAttr{}, ArrayAttr{}, operands);
      } else {
        rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, callee,
                                             ArrayAttr{}, ArrayAttr{}, operands);
        rewriter.eraseOp(op);
      }
    }
    return success();
  }

  std::string callee;
};

struct PTODeclareTileMemRefToEmitC
    : public OpConversionPattern<mlir::pto::DeclareTileMemRefOp> {
  using OpConversionPattern<
      mlir::pto::DeclareTileMemRefOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareTileMemRefOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type convertedType = getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert declare_tile_memref result type");
    rewriter.replaceOp(op, makeEmitCOpaqueConstant(rewriter, op.getLoc(),
                                                   convertedType, "nullptr"));
    return success();
  }
};

struct PTODeclareGlobalToEmitC
    : public OpConversionPattern<mlir::pto::DeclareGlobalOp> {
  using OpConversionPattern<
      mlir::pto::DeclareGlobalOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareGlobalOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type convertedType = getTypeConverter()->convertType(op.getEntry().getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(
          op, "failed to convert declare_global result type");
    if (auto tvTy = dyn_cast<TensorViewType>(op.getEntry().getType())) {
      if (auto stridesAttr =
              op->getAttrOfType<DenseI64ArrayAttr>(kGlobalTensorStridesAttrName)) {
        auto strides = stridesAttr.asArrayRef();
        if (strides.size() == static_cast<size_t>(tvTy.getRank())) {
          convertedType = emitc::OpaqueType::get(
              rewriter.getContext(),
              getGlobalTensorTypeStringFromShapeAndStrides(
                  tvTy.getElementType(), tvTy.getShape(), strides));
        }
      }
    }
    auto var = rewriter.create<emitc::VariableOp>(
        op.getLoc(), convertedType,
        emitc::OpaqueAttr::get(rewriter.getContext(), ""));
    rewriter.replaceOp(op, var.getResult());
    return success();
  }
};

struct PTODeclareEventIdArrayToEmitC
    : public OpConversionPattern<mlir::pto::DeclareEventIdArrayOp> {
  using OpConversionPattern<
      mlir::pto::DeclareEventIdArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareEventIdArrayOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type arrayTy = getTypeConverter()->convertType(op.getArray().getType());
    if (!arrayTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to map declared eventid_array type");

    auto array = rewriter
                     .create<emitc::VariableOp>(
                         op.getLoc(), arrayTy,
                         emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                     .getResult();
    rewriter.replaceOp(op, array);
    return success();
  }
};

struct PTOEventIdArrayGetToEmitC
    : public OpConversionPattern<mlir::pto::EventIdArrayGetOp> {
  using OpConversionPattern<
      mlir::pto::EventIdArrayGetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::EventIdArrayGetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value array = peelUnrealized(adaptor.getArray());
    Value index = peelUnrealized(adaptor.getIndex());

    Type resultTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to map eventid_array get result type");

    auto load =
        rewriter.create<emitc::SubscriptOp>(op.getLoc(), resultTy, array, index);
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

struct PTOEventIdArraySetToEmitC
    : public OpConversionPattern<mlir::pto::EventIdArraySetOp> {
  using OpConversionPattern<
      mlir::pto::EventIdArraySetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::EventIdArraySetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value array = peelUnrealized(adaptor.getArray());
    Value index = peelUnrealized(adaptor.getIndex());
    Value value = peelUnrealized(adaptor.getValue());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__EVENTID_ARRAY_STORE",
        ArrayAttr{}, ArrayAttr{}, ValueRange{array, index, value});
    rewriter.eraseOp(op);
    return success();
  }
};

// pto.declare_local_array -> emitc.variable of !emitc.array<...>.
// Renders as `T a[D1][D2]...;` in the emitted C++.
struct PTODeclareLocalArrayToEmitC
    : public OpConversionPattern<mlir::pto::DeclareLocalArrayOp> {
  using OpConversionPattern<
      mlir::pto::DeclareLocalArrayOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::DeclareLocalArrayOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Type arrayTy = getTypeConverter()->convertType(op.getArray().getType());
    if (!arrayTy)
      return rewriter.notifyMatchFailure(op,
                                         "failed to map !pto.local_array type");

    auto var = rewriter
                   .create<emitc::VariableOp>(
                       op.getLoc(), arrayTy,
                       emitc::OpaqueAttr::get(rewriter.getContext(), ""))
                   .getResult();
    rewriter.replaceOp(op, var);
    return success();
  }
};

// pto.local_array_get %a[%i0, %i1, ...] -> scalar snapshot.
// Materialize the subscript read immediately so the MLIR SSA result keeps its
// value even if a later pto.local_array_set mutates the same backing array slot.
struct PTOLocalArrayGetToEmitC
    : public OpConversionPattern<mlir::pto::LocalArrayGetOp> {
  using OpConversionPattern<
      mlir::pto::LocalArrayGetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::LocalArrayGetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultTy =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(
          op, "failed to map local_array element type");

    Value array = peelUnrealized(adaptor.getArray());
    SmallVector<Value> indices;
    indices.reserve(adaptor.getIndices().size());
    for (Value index : adaptor.getIndices())
      indices.push_back(peelUnrealized(index));

    auto sub = rewriter.create<emitc::SubscriptOp>(
        op.getLoc(), resultTy, array, indices);
    auto snapshot =
        rewriter
            .create<emitc::VariableOp>(
                op.getLoc(), resultTy,
                emitc::OpaqueAttr::get(rewriter.getContext(), ""))
            .getResult();
    rewriter.create<emitc::AssignOp>(op.getLoc(), snapshot, sub.getResult());
    rewriter.replaceOp(op, snapshot);
    return success();
  }
};

// pto.local_array_set %a[%i0, %i1, ...], %v -> emitc.assign to subscript slot.
// The C++ emitter prints this as `a[i0][i1]... = v;`. As above, adaptor values
// are already target-typed; pass them through directly.
struct PTOLocalArraySetToEmitC
    : public OpConversionPattern<mlir::pto::LocalArraySetOp> {
  using OpConversionPattern<
      mlir::pto::LocalArraySetOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::LocalArraySetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value value = adaptor.getValue();
    Type elemTy = value.getType();

    Value slot = rewriter
                     .create<emitc::SubscriptOp>(op.getLoc(), elemTy,
                                                 adaptor.getArray(),
                                                 adaptor.getIndices())
                     .getResult();
    rewriter.create<emitc::AssignOp>(op.getLoc(), slot, value);
    rewriter.eraseOp(op);
    return success();
  }
};

static std::optional<int64_t> getStaticIndexLikeValue(Value value) {
  if (!value)
    return std::nullopt;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

static FailureOr<Value> buildGlobalTensorViewFromPointer(
    ConversionPatternRewriter &rewriter, Location loc, Value ptr, Type elemTy,
    ArrayRef<int64_t> shape, ArrayRef<int64_t> strides = {},
    StringRef layoutEnum = "pto::Layout::ND") {
  if (llvm::any_of(shape, [](int64_t dim) {
        return dim == ShapedType::kDynamic;
      }))
    return failure();

  auto *ctx = rewriter.getContext();
  SmallVector<int64_t> rowMajorStrides;
  ArrayRef<int64_t> effectiveStrides = strides;
  if (effectiveStrides.empty()) {
    rowMajorStrides = buildRowMajorStrides(shape);
    effectiveStrides = rowMajorStrides;
  }
  SmallVector<int64_t, 5> shape5D;
  SmallVector<int64_t, 5> stride5D;
  buildGlobalTensorShapeAndStride(shape, effectiveStrides, shape5D, stride5D);

  std::string shapeType = "pto::Shape<" + joinIntTemplateParams(shape5D) + ">";
  std::string strideType =
      "pto::Stride<" + joinIntTemplateParams(stride5D) + ">";
  auto shapeVal = rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, emitc::OpaqueType::get(ctx, shapeType),
                          shapeType, ArrayAttr{}, ArrayAttr{}, ValueRange{})
                      .getResult(0);
  auto strideVal = rewriter
                       .create<emitc::CallOpaqueOp>(
                           loc, emitc::OpaqueType::get(ctx, strideType),
                           strideType, ArrayAttr{}, ArrayAttr{}, ValueRange{})
                       .getResult(0);

  std::string gtTypeStr =
      getGlobalTensorTypeStringFromShapeAndStrides(elemTy, shape,
                                                   effectiveStrides,
                                                   layoutEnum);
  auto gtType = emitc::OpaqueType::get(ctx, gtTypeStr);
  auto gt = rewriter.create<emitc::CallOpaqueOp>(
      loc, gtType, gtTypeStr, ArrayAttr{}, ArrayAttr{},
      ValueRange{ptr, shapeVal, strideVal});
  return gt.getResult(0);
}

static bool parseIntegerTemplateList(StringRef token, StringRef marker,
                                     SmallVectorImpl<int64_t> &values) {
  size_t pos = token.find(marker);
  if (pos == StringRef::npos)
    return false;
  pos += marker.size();
  size_t end = token.find('>', pos);
  if (end == StringRef::npos)
    return false;

  SmallVector<StringRef, 8> parts;
  token.slice(pos, end).split(parts, ',');
  values.clear();
  for (StringRef part : parts) {
    int64_t value = 0;
    if (part.trim().getAsInteger(10, value))
      return false;
    values.push_back(value);
  }
  return true;
}

static LogicalResult getStaticTensorViewStrides(
    Value source, Value convertedSource, pto::TensorViewType sourceType,
    SmallVectorImpl<int64_t> &strides) {
  int64_t rank = sourceType.getRank();
  strides.clear();

  if (auto makeView = source.getDefiningOp<pto::MakeTensorViewOp>()) {
    if ((int64_t)makeView.getStrides().size() != rank)
      return failure();
    for (Value strideValue : makeView.getStrides()) {
      auto cst = getStaticIndexLikeValue(strideValue);
      if (!cst)
        return failure();
      strides.push_back(*cst);
    }
    return success();
  }

  Value src = peelUnrealized(convertedSource);
  if (auto opaqueTy = dyn_cast<emitc::OpaqueType>(src.getType())) {
    SmallVector<int64_t, 5> stride5D;
    StringRef token = opaqueTy.getValue();
    if ((parseIntegerTemplateList(token, "pto::Stride<", stride5D) ||
         parseIntegerTemplateList(token, "Stride<", stride5D)) &&
        (int64_t)stride5D.size() >= rank) {
      strides.append(stride5D.end() - rank, stride5D.end());
      return success();
    }
  }

  auto fallback = buildRowMajorStrides(sourceType.getShape());
  strides.append(fallback.begin(), fallback.end());
  return success();
}

struct PTOPartitionViewToEmitC
    : public OpConversionPattern<mlir::pto::PartitionViewOp> {
  using OpConversionPattern<
      mlir::pto::PartitionViewOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::PartitionViewOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<pto::TensorViewType>(op.getSource().getType());
    auto resTy = dyn_cast<pto::PartitionTensorViewType>(op.getResult().getType());
    if (!srcTy || !resTy)
      return rewriter.notifyMatchFailure(
          op, "expected tensor_view source and partition_tensor_view result");

    if (op.getOffsets().size() != static_cast<size_t>(srcTy.getRank()) ||
        op.getSizes().size() != static_cast<size_t>(srcTy.getRank()))
      return rewriter.notifyMatchFailure(op, "rank mismatch");

    for (auto [idx, value] : llvm::enumerate(op.getSizes())) {
      auto cst = getStaticIndexLikeValue(value);
      if (!cst)
        return rewriter.notifyMatchFailure(
            op, "globaltensor partition_view requires static sizes");
      int64_t resultDim = resTy.getShape()[idx];
      if (resultDim != ShapedType::kDynamic && resultDim != *cst)
        return rewriter.notifyMatchFailure(
            op, "partition_view static size does not match result type");
    }

    SmallVector<int64_t> srcStrides;
    if (failed(getStaticTensorViewStrides(op.getSource(), adaptor.getSource(),
                                          srcTy, srcStrides)))
      return rewriter.notifyMatchFailure(
          op, "partition_view requires static source strides");
    int64_t staticLinearOffset = 0;
    SmallVector<std::pair<Value, int64_t>> dynamicOffsetTerms;
    for (auto [idx, values] :
         llvm::enumerate(llvm::zip(op.getOffsets(), adaptor.getOffsets()))) {
      Value originalOffset = std::get<0>(values);
      Value convertedOffset = std::get<1>(values);
      int64_t stride = srcStrides[idx];
      if (stride == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(
            op, "dynamic source stride is not supported");

      if (auto cst = getStaticIndexLikeValue(originalOffset)) {
        if (*cst != 0)
          staticLinearOffset += (*cst) * stride;
        continue;
      }
      dynamicOffsetTerms.push_back({convertedOffset, stride});
    }

    auto *ctx = rewriter.getContext();
    std::string elemTypeStr = getElemTypeStringForGT(srcTy.getElementType());
    auto ptrTy = emitc::PointerType::get(
        emitc::OpaqueType::get(ctx, "__gm__ " + elemTypeStr));
    Value src = peelUnrealized(adaptor.getSource());
    auto data = rewriter
                    .create<emitc::CallOpaqueOp>(
                        op.getLoc(), ptrTy, "PTOAS__GLOBAL_TENSOR_DATA",
                        ArrayAttr{}, ArrayAttr{}, ValueRange{src})
                    .getResult(0);
    Value ptr = data;
    if (!dynamicOffsetTerms.empty()) {
      Type u32Ty = emitc::OpaqueType::get(ctx, "unsigned");
      auto makeU32 = [&](int64_t value) {
        return makeEmitCIntConstant(rewriter, op.getLoc(), u32Ty, value);
      };
      auto asU32 = [&](Value value) -> Value {
        if (value.getType() == u32Ty)
          return value;
        return rewriter.create<emitc::CastOp>(op.getLoc(), u32Ty, value)
            .getResult();
      };

      Value totalOffset = makeU32(staticLinearOffset);
      for (auto [offsetValue, stride] : dynamicOffsetTerms) {
        Value term = asU32(offsetValue);
        if (stride != 1) {
          Value strideValue = makeU32(stride);
          term = rewriter
                     .create<emitc::MulOp>(op.getLoc(), u32Ty, term,
                                           strideValue)
                     .getResult();
        }
        totalOffset = rewriter
                          .create<emitc::AddOp>(op.getLoc(), u32Ty,
                                                totalOffset, term)
                          .getResult();
      }
      ptr = rewriter
                .create<emitc::AddOp>(op.getLoc(), data.getType(), data,
                                      totalOffset)
                .getResult();
    } else {
      ptr = applyStaticMemrefOffset(rewriter, op.getLoc(), data,
                                    staticLinearOffset);
    }

    auto resultOr = buildGlobalTensorViewFromPointer(
        rewriter, op.getLoc(), ptr, resTy.getElementType(), resTy.getShape(),
        srcStrides);
    if (failed(resultOr))
      return rewriter.notifyMatchFailure(
          op, "failed to materialize partition GlobalTensor");

    rewriter.replaceOp(op, *resultOr);
    return success();
  }
};

static FailureOr<std::string> getPipeDataTypeToken(Value value) {
  auto opaqueTy = dyn_cast<emitc::OpaqueType>(value.getType());
  if (!opaqueTy)
    return failure();
  StringRef token = opaqueTy.getValue();
  if (!token.contains("Tile<") && !token.contains("GlobalTensor<"))
    return failure();
  return token.str();
}

struct PTOTAllocToEmitC : public OpConversionPattern<mlir::pto::TAllocOp> {
  PTOTAllocToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                   PTOArch targetArch)
      : OpConversionPattern<mlir::pto::TAllocOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::TAllocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto pipeTok = getTPipeTokenFromValue(op.getPipeHandle(), targetArch);
    if (failed(pipeTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve pipe token");
    Value entry = peelUnrealized(adaptor.getEntry());
    auto entryTok = getPipeDataTypeToken(entry);
    if (failed(entryTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve entry token");
    auto splitTok = getTileSplitToken(op.getSplit());
    if (failed(splitTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve split token");

    std::string callee =
        "TALLOC<" + *pipeTok + ", " + *entryTok + ", " + *splitTok + ">";
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{peelUnrealized(adaptor.getPipeHandle()), entry});
    return success();
  }

  PTOArch targetArch;
};

struct PTOTPushToEmitC : public OpConversionPattern<mlir::pto::TPushOp> {
  PTOTPushToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                  PTOArch targetArch)
      : OpConversionPattern<mlir::pto::TPushOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::TPushOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto pipeTok = getTPipeTokenFromValue(op.getPipeHandle(), targetArch);
    if (failed(pipeTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve pipe token");
    // Read the tile type token from the already-converted OpaqueType, which
    // preserves the exact layout produced by BindTileOp / PointerCastOp EmitC.
    Value convertedTile = peelUnrealized(adaptor.getTile());
    auto tileTok = getPipeDataTypeToken(convertedTile);
    if (failed(tileTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve tile token");
    auto splitTok = getTileSplitToken(op.getSplit());
    if (failed(splitTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve split token");

    std::string callee =
        "TPUSH<" + *pipeTok + ", " + *tileTok + ", " + *splitTok + ">";
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{peelUnrealized(adaptor.getPipeHandle()), convertedTile});
    return success();
  }

  PTOArch targetArch;
};

struct PTOTPopToEmitC : public OpConversionPattern<mlir::pto::TPopOp> {
  PTOTPopToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                 PTOArch targetArch)
      : OpConversionPattern<mlir::pto::TPopOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::TPopOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto pipeTok = getTPipeTokenFromValue(op.getPipeHandle(), targetArch);
    if (failed(pipeTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve pipe token");
    Value convertedTile = peelUnrealized(adaptor.getTile());
    auto tileTok = getPipeDataTypeToken(convertedTile);
    if (failed(tileTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve tile token");
    auto splitTok = getTileSplitToken(op.getSplit());
    if (failed(splitTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve split token");

    std::string callee =
        "TPOP<" + *pipeTok + ", " + *tileTok + ", " + *splitTok + ">";
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee, ArrayAttr{}, ArrayAttr{},
        ValueRange{peelUnrealized(adaptor.getPipeHandle()), convertedTile});
    return success();
  }

  PTOArch targetArch;
};

struct PTOTFreeToEmitC : public OpConversionPattern<mlir::pto::TFreeOp> {
  PTOTFreeToEmitC(TypeConverter &typeConverter, MLIRContext *ctx,
                  PTOArch targetArch)
      : OpConversionPattern<mlir::pto::TFreeOp>(typeConverter, ctx),
        targetArch(targetArch) {}

  LogicalResult matchAndRewrite(mlir::pto::TFreeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto pipeTok = getTPipeTokenFromValue(op.getPipeHandle(), targetArch);
    if (failed(pipeTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve pipe token");
    auto splitTok = getTileSplitToken(op.getSplit());
    if (failed(splitTok))
      return rewriter.notifyMatchFailure(op, "failed to resolve split token");

    SmallVector<Value> operands{peelUnrealized(adaptor.getPipeHandle())};
    std::string callee;
    if (op.getEntry()) {
      Value entry = peelUnrealized(adaptor.getEntry());
      auto entryTok = getPipeDataTypeToken(entry);
      if (failed(entryTok))
        return rewriter.notifyMatchFailure(op, "failed to resolve entry token");
      callee = "TFREE<" + *pipeTok + ", " + *entryTok + ", " + *splitTok + ">";
      operands.push_back(entry);
    } else {
      callee = "TFREE<" + *pipeTok + ", " + *splitTok + ">";
    }
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, callee, ArrayAttr{}, ArrayAttr{}, operands);
    return success();
  }

  PTOArch targetArch;
};

//===----------------------------------------------------------------------===//
// populate patterns
//===----------------------------------------------------------------------===
struct ReinterpretCastToEmitC : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern<memref::ReinterpretCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    auto resMrTy = dyn_cast<MemRefType>(op.getType());
    if (!resMrTy)
      return failure();

    auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(resMrTy.getMemorySpace());
    const bool isGm = (!asAttr || asAttr.getAddressSpace() == pto::AddressSpace::GM);

    bool emitAddPtrTrace = op->hasAttr("pto.addptr_trace");
    Value source = peelUnrealized(adaptor.getSource());
    auto offsets = adaptor.getOffsets();
    Value offsetVal = offsets.empty() ? Value() : offsets[0];

    // GM: keep pointer arithmetic.
    if (isGm) {
      if (!offsetVal) {
        rewriter.replaceOp(op, source);
        return success();
      }

      Type resultType = getTypeConverter()->convertType(op.getType());
      if (!resultType)
        return failure();

      auto addOp = rewriter.create<emitc::AddOp>(loc, resultType, source, offsetVal);
      if (emitAddPtrTrace) {
        rewriter.setInsertionPointAfter(addOp);
        rewriter.create<emitc::CallOpaqueOp>(
            loc, TypeRange{}, "PTOAS__ADDPTR_TRACE",
            ArrayAttr{}, ArrayAttr{},
            ValueRange{addOp.getResult(), source, offsetVal});
      }
      rewriter.replaceOp(op, addOp.getResult());
      return success();
    }

    // UB/L1/L0 tiles: materialize a new Tile view by assigning an adjusted
    // underlying pointer (in elements).
    pto::AddressSpace as = asAttr.getAddressSpace();

    // Element type token.
    Type elemTy = resMrTy.getElementType();
    std::string elemTok = getEmitCScalarTypeToken(elemTy);
    int64_t elemBytes = getEmitCScalarByteWidth(elemTy);

    // Tile role.
    const char *roleTok = "TileType::Vec";
    switch (as) {
    case pto::AddressSpace::VEC:
      roleTok = "TileType::Vec";
      break;
    case pto::AddressSpace::MAT:
      roleTok = "TileType::Mat";
      break;
    case pto::AddressSpace::LEFT:
      roleTok = "TileType::Left";
      break;
    case pto::AddressSpace::RIGHT:
      roleTok = "TileType::Right";
      break;
    case pto::AddressSpace::ACC:
      roleTok = "TileType::Acc";
      break;
    case pto::AddressSpace::BIAS:
      roleTok = "TileType::Bias";
      break;
    case pto::AddressSpace::GM:
      roleTok = "TileType::Vec";
      break;
    case pto::AddressSpace::Zero:
      roleTok = "TileType::Vec";
      break;
    case pto::AddressSpace::SCALING:
      roleTok = "TileType::Scaling";
      break;
    }

    // Shape (fallback to 32x32).
    int64_t rows = 32, cols = 32;
    if (resMrTy.getRank() >= 2 && resMrTy.hasStaticShape()) {
      rows = resMrTy.getDimSize(0);
      cols = resMrTy.getDimSize(1);
    }
    int64_t templateRows =
        renderTileTemplateDim(rows, elemTy, pto::BLayout::RowMajor, 0);
    int64_t templateCols =
        renderTileTemplateDim(cols, elemTy, pto::BLayout::RowMajor, 1);

    // Keep a conservative default config for now.
    std::string tileTypeStr =
        std::string("Tile<") + roleTok + ", " + elemTok + ", " +
        std::to_string(templateRows) + ", " + std::to_string(templateCols) +
        ", BLayout::RowMajor, " + std::to_string(templateRows) + ", " +
        std::to_string(templateCols) +
        ", SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>";

    auto tileType = emitc::OpaqueType::get(ctx, tileTypeStr);
    Value tile = rewriter
                     .create<emitc::VariableOp>(loc, tileType,
                                                emitc::OpaqueAttr::get(ctx, ""))
                     .getResult();

    // Compute an integer address and assign it to the new tile.
    // NOTE: pto-isa TASSIGN requires an integral address (not a pointer).
    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
    auto rcU64 = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});

    // Non-GM reinterpret_cast operands come from UB/L1/L0 tiles.
    // We need the underlying address, but `__cce_get_tile_ptr()` is only valid
    // inside `__tf__` functions. Use `tile.data()` (via a post-processed marker)
    // and compute the adjusted address in bytes.
    Value rawPtr = source;
    if (auto ot = dyn_cast<emitc::OpaqueType>(source.getType())) {
      // Only Tiles have a `.data()` member. For plain address-space pointers
      // (e.g. `__ubuf__ float*`), use the pointer value directly.
      if (ot.getValue().starts_with("Tile<")) {
        rawPtr = materializeTileDataValue(rewriter, loc, source, as, elemTok);
      }
    }

    Value baseAddr = rawPtr;
    if (isSetFFTsPointerLikeType(rawPtr.getType())) {
      baseAddr = rewriter
                     .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                                  /*args=*/ArrayAttr{},
                                                  /*templateArgs=*/rcU64,
                                                  /*operands=*/ValueRange{rawPtr})
                     .getResult(0);
    } else if (rawPtr.getType() != u64Ty) {
      baseAddr = rewriter.create<emitc::CastOp>(loc, u64Ty, rawPtr).getResult();
    }

    Value addr = baseAddr;
    if (offsetVal) {
      Value offU64 = offsetVal;
      if (offU64.getType() != u64Ty)
        offU64 = rewriter.create<emitc::CastOp>(loc, u64Ty, offU64).getResult();

      auto bytesAttr = emitc::OpaqueAttr::get(ctx, std::to_string(elemBytes));
      Value bytesVal = rewriter.create<emitc::ConstantOp>(loc, u64Ty, bytesAttr);
      Value byteOff = rewriter.create<emitc::MulOp>(loc, u64Ty, offU64, bytesVal);
      addr = rewriter.create<emitc::AddOp>(loc, u64Ty, baseAddr, byteOff);
    }

    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                         /*args=*/ArrayAttr{},
                                         /*templateArgs=*/ArrayAttr{},
                                         /*operands=*/ValueRange{tile, addr});

    rewriter.replaceOp(op, tile);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.taddc lowering -> TADDC(dst, src0, src1, src2)
//===----------------------------------------------------------------------===//

struct PTOTAddCToTADDC : public OpConversionPattern<pto::TAddCOp> {
  using OpConversionPattern<pto::TAddCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src2 = peelUnrealized(adaptor.getSrc2());
    Value dst  = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TADDC yet.
    // Decompose: dst = src0 + src1 + src2
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, dst, src2});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tadds lowering -> TADDS(dst, src, scalar)
//===----------------------------------------------------------------------===//

struct PTOAddSToTADDS : public OpConversionPattern<pto::TAddSOp> {
  using OpConversionPattern<pto::TAddSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = peelUnrealized(adaptor.getSrc());
    Value dst    = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TADDS", ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.taddsc lowering -> TADDSC(dst, src0, scalar, src1)
//===----------------------------------------------------------------------===//

struct PTOAddSCToTADDSC : public OpConversionPattern<pto::TAddSCOp> {
  using OpConversionPattern<pto::TAddSCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAddSCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0    = peelUnrealized(adaptor.getSrc0());
    Value scalar  = peelUnrealized(adaptor.getScalar());
    Value src1    = peelUnrealized(adaptor.getSrc1());
    Value dst     = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TADDSC yet.
    // Decompose: dst = src0 + scalar + src1
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, scalar});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, dst, src1});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOTAndToEmitC : public OpConversionPattern<pto::TAndOp> {
  using OpConversionPattern<pto::TAndOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAndOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a   = peelUnrealized(adaptor.getSrc0());
    Value b   = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TAND",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, a, b});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOConcatToEmitC : public OpConversionPattern<pto::TConcatOp> {
  using OpConversionPattern<pto::TConcatOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TConcatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TCONCAT",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOConcatidxToEmitC : public OpConversionPattern<pto::TConcatidxOp> {
  using OpConversionPattern<pto::TConcatidxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TConcatidxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src0Idx = peelUnrealized(adaptor.getSrc0Idx());
    Value src1Idx = peelUnrealized(adaptor.getSrc1Idx());
    Value dst  = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TCONCAT",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1, src0Idx, src1Idx});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOAndSToEmitC : public OpConversionPattern<pto::TAndSOp> {
  using OpConversionPattern<pto::TAndSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TAndSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TANDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};


struct PTOTCIToEmitC : public OpConversionPattern<pto::TCIOp> {
  using OpConversionPattern<pto::TCIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TCIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value S = peelUnrealized(adaptor.getOperands()[0]);

    // The TCI scalar template parameter should follow the original PTO IR
    // scalar type, not the converted EmitC value type.
    std::string scalarTok = "int32_t";
    if (auto it = dyn_cast<IntegerType>(op->getOperand(0).getType())) {
      bool isUnsigned = it.isUnsigned();
      if (it.getWidth() == 16)
        scalarTok = isUnsigned ? "uint16_t" : "int16_t";
      else
        scalarTok = isUnsigned ? "uint32_t" : "int32_t";
    }

    // descending -> "0"/"1"
    std::string descTok = op.getDescending() ? "1" : "0";

    ArrayAttr targs;
    if (auto ot = mlir::dyn_cast<emitc::OpaqueType>(dst.getType())) {
      std::string tileTok = ot.getValue().str(); // "Tile<...>"
      targs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, tileTok),
          emitc::OpaqueAttr::get(ctx, scalarTok),
          emitc::OpaqueAttr::get(ctx, descTok),
      });
    } else {
      targs = rewriter.getArrayAttr({});
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCI",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/targs,
        /*operands=*/ValueRange{dst, S});

    rewriter.eraseOp(op);
    return success();
  }
};
static std::string cmpModeTok(pto::CmpModeAttr a) {
  // 生成 "CmpMode::GT" 这种 token
  auto m = a.getValue(); // 取 enum
  switch (m) {
    case pto::CmpMode::EQ: return "CmpMode::EQ";
    case pto::CmpMode::NE: return "CmpMode::NE";
    case pto::CmpMode::LT: return "CmpMode::LT";
    case pto::CmpMode::LE: return "CmpMode::LE";
    case pto::CmpMode::GT: return "CmpMode::GT";
    case pto::CmpMode::GE: return "CmpMode::GE";
  }
  return "CmpMode::EQ";
}
struct PTOColExpandToEmitC : public OpConversionPattern<pto::TColExpandOp> {
  using OpConversionPattern<pto::TColExpandOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPAND",
        /*args=*/ArrayAttr(),           
        /*templateArgs=*/ArrayAttr(),
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandMulToEmitC : public OpConversionPattern<pto::TColExpandMulOp> {
  using OpConversionPattern<pto::TColExpandMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDMUL",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandAddToEmitC : public OpConversionPattern<pto::TColExpandAddOp> {
  using OpConversionPattern<pto::TColExpandAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDADD",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandDivToEmitC : public OpConversionPattern<pto::TColExpandDivOp> {
  using OpConversionPattern<pto::TColExpandDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    ArrayAttr templateArgs;
    if (op.getPrecisionType() != pto::DivPrecision::Default) {
      StringRef precisionTok;
      switch (op.getPrecisionType()) {
      case pto::DivPrecision::Default:
        precisionTok = "pto::DivAlgorithm::DEFAULT";
        break;
      case pto::DivPrecision::HighPrecision:
        precisionTok = "pto::DivAlgorithm::HIGH_PRECISION";
        break;
      }
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, precisionTok)});
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDDIV",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/templateArgs,
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandExpdifToEmitC
    : public OpConversionPattern<pto::TColExpandExpdifOp> {
  using OpConversionPattern<pto::TColExpandExpdifOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandExpdifOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDEXPDIF",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandSubToEmitC : public OpConversionPattern<pto::TColExpandSubOp> {
  using OpConversionPattern<pto::TColExpandSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDSUB",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandMaxToEmitC : public OpConversionPattern<pto::TColExpandMaxOp> {
  using OpConversionPattern<pto::TColExpandMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDMAX",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColExpandMinToEmitC : public OpConversionPattern<pto::TColExpandMinOp> {
  using OpConversionPattern<pto::TColExpandMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColExpandMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPANDMIN",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOTTriToEmitC : public OpConversionPattern<pto::TTriOp> {
  using OpConversionPattern<pto::TTriOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TTriOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value diagonal = peelUnrealized(adaptor.getDiagonal());

    ArrayAttr templateArgs;
    if (auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType())) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, std::to_string(op.getUpperOrLower())),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    SmallVector<Value, 2> operands{dst, diagonal};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TTRI",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs, operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOCmpToEmitC : public OpConversionPattern<pto::TCmpOp> {
  using OpConversionPattern<pto::TCmpOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TCmpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
	
    Value dst  = peelUnrealized(adaptor.getDst());
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());

    std::string tok = "CmpMode::EQ";
    if (auto a = op.getCmpModeAttr())
      tok = cmpModeTok(a);

    auto modeTy = emitc::OpaqueType::get(ctx, "CmpMode");
    Value modeVal = rewriter.create<emitc::ConstantOp>(
        loc, modeTy, emitc::OpaqueAttr::get(ctx, tok));

    rewriter.create<emitc::CallOpaqueOp>(
        loc,
        TypeRange{},
        "TCMP",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1, modeVal});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOCmpSToEmitC : public OpConversionPattern<pto::TCmpSOp> {
  using OpConversionPattern<pto::TCmpSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TCmpSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());

    // cmpMode -> token
    auto cmpAttr = op.getCmpModeAttr();          // PTO_CmpModeAttr
    std::string tok = cmpModeTok(cmpAttr);

    auto modeTy = emitc::OpaqueType::get(ctx, "CmpMode");
    Value modeVal = rewriter.create<emitc::ConstantOp>(
        loc, modeTy, emitc::OpaqueAttr::get(ctx, tok));

    rewriter.create<emitc::CallOpaqueOp>(
        loc,
        TypeRange{},
        "TCMPS",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, scalar, modeVal});

    rewriter.eraseOp(op);
    return success();
  }
};


struct PTOColMaxToEmitC : public OpConversionPattern<pto::TColMaxOp> {
  using OpConversionPattern<pto::TColMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // intrinsic: TCOLMAX(dst, src)
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TCOLMAX", ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColArgMaxToEmitC : public OpConversionPattern<pto::TColArgMaxOp> {
  using OpConversionPattern<pto::TColArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLARGMAX",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColMinToEmitC : public OpConversionPattern<pto::TColMinOp> {
  using OpConversionPattern<pto::TColMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // intrinsic: TCOLMIN(dst, src)
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TCOLMIN", ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColArgMinToEmitC : public OpConversionPattern<pto::TColArgMinOp> {
  using OpConversionPattern<pto::TColArgMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColArgMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLARGMIN",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColSumToEmitC : public OpConversionPattern<pto::TColSumOp> {
  using OpConversionPattern<pto::TColSumOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColSumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // Check if tmp exists before accessing it
    if (op.getTmp()) {
      // Format 2: with tmp and isBinary
      Value tmp = peelUnrealized(adaptor.getTmp());
      bool isBinary = false;
      if (auto a = op.getIsBinaryAttr())
        isBinary = a.getValue();

      auto boolTy = emitc::OpaqueType::get(ctx, "bool");
      auto tok = isBinary ? "true" : "false";
      Value isBinaryVal = rewriter.create<emitc::ConstantOp>(
          loc, boolTy, emitc::OpaqueAttr::get(ctx, tok));

      SmallVector<unsigned, 3> tileSlotOrder;
      tileSlotOrder.push_back(op.getDstMutable().getOperandNumber());
      tileSlotOrder.push_back(op.getSrcMutable().getOperandNumber());
      tileSlotOrder.push_back(op.getTmpMutable().begin()->getOperandNumber());

      createLastUseAwareOpaqueCall(
          rewriter, op.getOperation(), TypeRange{}, "TCOLSUM",
          ValueRange{dst, src, tmp, isBinaryVal}, ArrayAttr{}, ArrayAttr{},
          tileSlotOrder);
    } else {
      // Format 1: without tmp and isBinary
      SmallVector<unsigned, 2> tileSlotOrder;
      tileSlotOrder.push_back(op.getDstMutable().getOperandNumber());
      tileSlotOrder.push_back(op.getSrcMutable().getOperandNumber());
      createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                   "TCOLSUM", ValueRange{dst, src},
                                   ArrayAttr{}, ArrayAttr{}, tileSlotOrder);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOColProdToEmitC : public OpConversionPattern<pto::TColProdOp> {
  using OpConversionPattern<pto::TColProdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TColProdOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLPROD",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
static std::string roundModeTok(mlir::pto::RoundModeAttr attr) {
  using RM = mlir::pto::RoundMode;
  switch (attr.getValue()) {
  case RM::NONE:      return "RoundMode::CAST_NONE";
  case RM::RINT:      return "RoundMode::CAST_RINT";
  case RM::ROUND:     return "RoundMode::CAST_ROUND";
  case RM::FLOOR:     return "RoundMode::CAST_FLOOR";
  case RM::CEIL:      return "RoundMode::CAST_CEIL";
  case RM::TRUNC:     return "RoundMode::CAST_TRUNC";
  case RM::ODD:       return "RoundMode::CAST_ODD";
  case RM::CAST_RINT: return "RoundMode::CAST_RINT";
  }
  return "RoundMode::CAST_RINT";
}
static std::string saturationModeTok(mlir::pto::SaturationModeAttr attr) {
  using SM = mlir::pto::SaturationMode;
  switch (attr.getValue()) {
  case SM::ON:  return "SaturationMode::ON";
  case SM::OFF: return "SaturationMode::OFF";
  }
  return "SaturationMode::OFF";
}
struct PTOCvtToEmitC : public OpConversionPattern<pto::TCvtOp> {
  using OpConversionPattern<pto::TCvtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TCvtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    pto::RoundModeAttr rmAttr = op.getRmodeAttr();
    std::string rmTok = rmAttr ? roundModeTok(rmAttr)
                               : std::string("RoundMode::CAST_RINT");
    auto rmodeTy = emitc::OpaqueType::get(ctx, "RoundMode");
    Value rmodeVal = rewriter.create<emitc::ConstantOp>(
        loc, rmodeTy, emitc::OpaqueAttr::get(ctx, rmTok));

    auto satModeTy = emitc::OpaqueType::get(ctx, "SaturationMode");
    auto satAttr = op.getSatModeAttr();
    std::string satTok = satAttr ? saturationModeTok(satAttr)
                                 : std::string("SaturationMode::OFF");
    Value satModeVal = rewriter.create<emitc::ConstantOp>(
        loc, satModeTy, emitc::OpaqueAttr::get(ctx, satTok));

    SmallVector<Value, 4> operands{dst, src, rmodeVal, satModeVal};

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCVT",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTORandomToEmitC : public OpConversionPattern<pto::TRandomOp> {
  using OpConversionPattern<pto::TRandomOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRandomOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    SmallVector<Value, 7> operands{
        dst,
        peelUnrealized(adaptor.getKey0()),
        peelUnrealized(adaptor.getKey1()),
        peelUnrealized(adaptor.getCounter0()),
        peelUnrealized(adaptor.getCounter1()),
        peelUnrealized(adaptor.getCounter2()),
        peelUnrealized(adaptor.getCounter3()),
    };
    ArrayAttr templateArgs = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(ctx, std::to_string(op.getRounds()))});

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "PTOAS__TRANDOM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs, operands);
    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tdiv lowering -> TDIV(dst, src0, src1)
//===----------------------------------------------------------------------===//

struct PTODivToTDIV : public OpConversionPattern<pto::TDivOp> {
  using OpConversionPattern<pto::TDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());
    auto *ctx = rewriter.getContext();

    ArrayAttr templateArgs;
    if (op.getPrecisionType() != pto::DivPrecision::Default) {
      StringRef precisionTok;
      switch (op.getPrecisionType()) {
      case pto::DivPrecision::Default:
        precisionTok = "pto::DivAlgorithm::DEFAULT";
        break;
      case pto::DivPrecision::HighPrecision:
        precisionTok = "pto::DivAlgorithm::HIGH_PRECISION";
        break;
      }
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, precisionTok)});
    }

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TDIV", ValueRange{dst, src0, src1}, ArrayAttr{}, templateArgs);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tdivs lowering -> TDIVS(dst, src, scalar)  or  TDIVS(dst, scalar, src)
// Order is determined by operand types: if src is tile_buf, order is (tile, scalar)
// Otherwise, order is (scalar, tile)
//===----------------------------------------------------------------------===//

struct PTODivSToEmitC : public OpConversionPattern<pto::TDivSOp> {
  using OpConversionPattern<pto::TDivSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDivSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());
    // Preserve source order from textual parse:
    // ins(tile, scalar)   -> TDIVS(dst, tile, scalar)
    // ins(scalar, tile)   -> TDIVS(dst, scalar, tile)
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TDIVS", ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.tdivs (TDivSOp) lowering -> TDIVS(dst, src, scalar)  or  TDIVS(dst, scalar, src)
// Order is determined by operand types: if src is tile_buf, order is (tile, scalar)
// Otherwise, order is (scalar, tile)
//===----------------------------------------------------------------------===//

struct PTOTDivSToEmitC : public OpConversionPattern<pto::TDivSOp> {
  using OpConversionPattern<pto::TDivSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDivSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TDIVS", ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.texp lowering -> TEXP(dst, src)
//===----------------------------------------------------------------------===//

struct PTOExpToEmitC : public OpConversionPattern<pto::TExpOp> {
  using OpConversionPattern<pto::TExpOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    ArrayAttr templateArgs;
    if (op.getPrecisionType() != pto::ExpPrecision::Default) {
      StringRef precisionTok;
      switch (op.getPrecisionType()) {
      case pto::ExpPrecision::Default:
        precisionTok = "pto::ExpAlgorithm::DEFAULT";
        break;
      case pto::ExpPrecision::HighPrecision:
        precisionTok = "pto::ExpAlgorithm::HIGH_PRECISION";
        break;
      }
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, precisionTok)});
    }

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TEXP", ValueRange{dst, src}, ArrayAttr{}, templateArgs);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.texpands lowering -> TEXPANDS(dst, scalar)
//===----------------------------------------------------------------------===//

struct PTOExpandsToEmitC : public OpConversionPattern<pto::TExpandsOp> {
  using OpConversionPattern<pto::TExpandsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExpandsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TEXPANDS", ValueRange{dst, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.textract lowering -> TEXTRACT(dst, src, indexRow, indexCol)
//===----------------------------------------------------------------------===//

struct PTOExtractToEmitC : public OpConversionPattern<pto::TExtractOp> {
  using OpConversionPattern<pto::TExtractOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExtractOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value r0  = peelUnrealized(adaptor.getIndexRow());
    Value c0  = peelUnrealized(adaptor.getIndexCol());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TEXTRACT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, r0, c0});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.textract_fp lowering -> TEXTRACT_FP(dst, src, fp, indexRow, indexCol)
//===----------------------------------------------------------------------===//

struct PTOExtractFPToEmitC : public OpConversionPattern<pto::TExtractFPOp> {
  using OpConversionPattern<pto::TExtractFPOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TExtractFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value fp = peelUnrealized(adaptor.getFp());
    Value dst = peelUnrealized(adaptor.getDst());
    Value r0 = peelUnrealized(adaptor.getIndexRow());
    Value c0 = peelUnrealized(adaptor.getIndexCol());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TEXTRACT_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, fp, r0, c0});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tinsert lowering -> TINSERT(dst, src, indexRow, indexCol)
// Keep lowering arch-agnostic and let PTO-ISA infer proper A5 path.
//===----------------------------------------------------------------------===//

struct PTOInsertToEmitC : public OpConversionPattern<pto::TInsertOp> {
  using OpConversionPattern<pto::TInsertOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TInsertOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value r0  = peelUnrealized(adaptor.getIndexRow());
    Value c0  = peelUnrealized(adaptor.getIndexCol());
    Value fp;
    if (op.getFp())
      fp = peelUnrealized(adaptor.getFp());
    Value preQuantScalar;
    if (op.getPreQuantScalar())
      preQuantScalar = peelUnrealized(adaptor.getPreQuantScalar());

    auto reluTok = [&](pto::ReluPreMode mode) -> StringRef {
      switch (mode) {
      case pto::ReluPreMode::NoRelu:     return "ReluPreMode::NoRelu";
      case pto::ReluPreMode::NormalRelu: return "ReluPreMode::NormalRelu";
      case pto::ReluPreMode::ScalarRelu: return "ReluPreMode::ScalarRelu";
      case pto::ReluPreMode::VectorRelu: return "ReluPreMode::VectorRelu";
      case pto::ReluPreMode::Pwl:        return "ReluPreMode::Pwl";
      }
      llvm_unreachable("unknown ReluPreMode");
    };

    auto accToVecModeTok = [&](pto::AccToVecMode mode) -> StringRef {
      switch (mode) {
      case pto::AccToVecMode::SingleModeVec0: return "pto::AccToVecMode::SingleModeVec0";
      case pto::AccToVecMode::SingleModeVec1: return "pto::AccToVecMode::SingleModeVec1";
      case pto::AccToVecMode::DualModeSplitM: return "pto::AccToVecMode::DualModeSplitM";
      case pto::AccToVecMode::DualModeSplitN: return "pto::AccToVecMode::DualModeSplitN";
      }
      llvm_unreachable("unknown AccToVecMode");
    };

    auto tinsertModeTok = [&](pto::TInsertMode mode) -> StringRef {
      switch (mode) {
      case pto::TInsertMode::SPLIT2: return "pto::TInsertMode::SPLIT2";
      case pto::TInsertMode::SPLIT4: return "pto::TInsertMode::SPLIT4";
      }
      llvm_unreachable("unknown TInsertMode");
    };

    auto tinsertModeAttr = op.getTinsertModeAttr();
    auto accToVecModeAttr = op.getAccToVecModeAttr();
    const bool hasFp = static_cast<bool>(fp);
    const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);
    const bool hasTinsertMode = static_cast<bool>(tinsertModeAttr);
    const bool hasAccToVecMode = static_cast<bool>(accToVecModeAttr);
    const bool reluNonDefault = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
    const bool needTemplateArgs = hasFp || hasPreQuantScalar || hasTinsertMode ||
                                  hasAccToVecMode || reluNonDefault;

    SmallVector<Value> operands{dst, src, r0, c0};
    if (fp)
      operands.push_back(fp);
    if (preQuantScalar)
      operands.push_back(preQuantScalar);

    ArrayAttr templateArgs = ArrayAttr{};
    if (needTemplateArgs) {
      auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
      auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
      if (!dstOT || !srcOT)
        return rewriter.notifyMatchFailure(
            op, "tinsert lowering expects opaque dst/src types");

      SmallVector<Attribute, 6> templateArgVec;
      if (hasTinsertMode)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, tinsertModeTok(tinsertModeAttr.getValue())));
      templateArgVec.push_back(emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()));
      templateArgVec.push_back(emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()));
      if (hasFp) {
        auto fpOT = mlir::dyn_cast<emitc::OpaqueType>(fp.getType());
        if (!fpOT)
          return rewriter.notifyMatchFailure(
              op, "tinsert lowering expects opaque fp type");
        templateArgVec.push_back(emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()));
      }
      if (hasAccToVecMode)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, accToVecModeTok(accToVecModeAttr.getValue())));
      if (reluNonDefault)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, reluTok(op.getReluPreMode())));
      templateArgs = rewriter.getArrayAttr(templateArgVec);
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TINSERT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tinsert_fp lowering -> TINSERT_FP(dst, src, fp, indexRow, indexCol)
//===----------------------------------------------------------------------===//

struct PTOInsertFPToEmitC : public OpConversionPattern<pto::TInsertFPOp> {
  using OpConversionPattern<pto::TInsertFPOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TInsertFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value fp = peelUnrealized(adaptor.getFp());
    Value dst = peelUnrealized(adaptor.getDst());
    Value r0 = peelUnrealized(adaptor.getIndexRow());
    Value c0 = peelUnrealized(adaptor.getIndexCol());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TINSERT_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, fp, r0, c0});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tfillpad lowering -> TFILLPAD(dst, src)
//===----------------------------------------------------------------------===//

struct PTOFillPadToEmitC : public OpConversionPattern<pto::TFillPadOp> {
  using OpConversionPattern<pto::TFillPadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFillPadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFILLPAD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tfillpad_inplace lowering -> TFILLPAD_INPLACE(dst, src)
//===----------------------------------------------------------------------===//

struct PTOFillPadInplaceToEmitC
    : public OpConversionPattern<pto::TFillPadInplaceOp> {
  using OpConversionPattern<pto::TFillPadInplaceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFillPadInplaceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFILLPAD_INPLACE",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tfillpad_expand lowering -> TFILLPAD_EXPAND(dst, src)
//===----------------------------------------------------------------------===//

struct PTOFillPadExpandToEmitC
    : public OpConversionPattern<pto::TFillPadExpandOp> {
  using OpConversionPattern<pto::TFillPadExpandOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFillPadExpandOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFILLPAD_EXPAND",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tgather lowering
// - Index form  : TGATHER(dst, src0, indices, tmp)
// - Compare form: TGATHER<DstT, SrcT, CDstT, TmpT, CmpMode::GT, 7>(dst, src0, kValue, cdst, tmp)
// - Mask form : TGATHER<dstTileTok, srcTileTok, pto::MaskPattern::Pxxxx>(dst, src0)
//===----------------------------------------------------------------------===//

[[maybe_unused]] static std::string maskPatternTok(mlir::pto::MaskPatternAttr a) {

  auto v = a.getValue(); // enum
  return (std::string("pto::MaskPattern::") + mlir::pto::stringifyMaskPattern(v).str());
}

struct PTOGatherToEmitC : public OpConversionPattern<pto::TGatherOp> {
  using OpConversionPattern<pto::TGatherOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGatherOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst  = peelUnrealized(adaptor.getDst());
    Value src0 = peelUnrealized(adaptor.getSrc());

    auto getOpaqueTok = [&](Value v, StringRef name) -> FailureOr<std::string> {
      if (auto ot = mlir::dyn_cast<emitc::OpaqueType>(v.getType()))
        return ot.getValue().str();
      return rewriter.notifyMatchFailure(op, (name + " must be emitc::OpaqueType (tile)").str());
    };

    // Case 1: index-based TGATHER(dst, src0, indices, tmp)
    if (Value idx = adaptor.getIndices()) {
      idx = peelUnrealized(idx);
      Value tmp = peelUnrealized(adaptor.getTmp());

      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TGATHER",
          /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
          /*operands=*/ValueRange{dst, src0, idx, tmp});

      rewriter.eraseOp(op);
      return success();
    }

    // Case 2: compare-based TGATHER<DstT, SrcT, TmpT, CDstT, CmpMode::GT>(
    //            dst, src0, kValue, tmp, cdst, offset)
    if (Value cdst = adaptor.getCdst()) {
      cdst = peelUnrealized(cdst);
      Value tmp = peelUnrealized(adaptor.getTmp());
      Value kValue = peelUnrealized(adaptor.getKValue());

      auto dstTokOr = getOpaqueTok(dst, "dst");
      auto srcTokOr = getOpaqueTok(src0, "src0");
      auto cdstTokOr = getOpaqueTok(cdst, "cdst");
      auto tmpTokOr = getOpaqueTok(tmp, "tmp");
      if (failed(dstTokOr) || failed(srcTokOr) || failed(cdstTokOr) || failed(tmpTokOr))
        return failure();

      auto cmpAttr = op.getCmpModeAttr();
      std::string cmpTok = cmpAttr ? cmpModeTok(cmpAttr) : "CmpMode::EQ";
      int64_t offset = 0;
      if (auto offsetAttr = op.getOffsetAttr())
        offset = offsetAttr.getInt();
      auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
      Value offsetVal = makeEmitCIntConstant(rewriter, loc, i32Ty, offset);

      auto targs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, *dstTokOr),
          emitc::OpaqueAttr::get(ctx, *srcTokOr),
          emitc::OpaqueAttr::get(ctx, *tmpTokOr),
          emitc::OpaqueAttr::get(ctx, *cdstTokOr),
          emitc::OpaqueAttr::get(ctx, cmpTok),
      });

      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TGATHER",
          /*args=*/ArrayAttr{}, /*templateArgs=*/targs,
          /*operands=*/ValueRange{dst, src0, kValue, tmp, cdst, offsetVal});

      rewriter.eraseOp(op);
      return success();
    }

    // Case 3: mask-pattern TGATHER<DstT, SrcT, MaskPattern::P0101>(dst, src0)
    auto mp = op.getMaskPatternAttr();
    if (!mp)
      return rewriter.notifyMatchFailure(op, "expected maskPattern, indices, or cdst on tgather");

    auto dstTokOr = getOpaqueTok(dst, "dst");
    auto srcTokOr = getOpaqueTok(src0, "src0");
    if (failed(dstTokOr) || failed(srcTokOr))
      return failure();

    // mp is an EnumAttr; stringify name is "P0101" etc.
    // We emit MaskPattern::P0101 (because generated C++ has `using namespace pto;`)
    std::string mpTok = std::string("MaskPattern::") +
                        mlir::pto::stringifyMaskPattern(mp.getValue()).str();

    auto targs = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, *dstTokOr),
        emitc::OpaqueAttr::get(ctx, *srcTokOr),
        emitc::OpaqueAttr::get(ctx, mpTok),
    });

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGATHER",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/targs,
        /*operands=*/ValueRange{dst, src0});

    rewriter.eraseOp(op);
    return success();
  }
};


struct PTOGatherbToEmitC : public OpConversionPattern<pto::TGatherBOp> {
  using OpConversionPattern<pto::TGatherBOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGatherBOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src     = peelUnrealized(adaptor.getSrc());
    Value offsets = peelUnrealized(adaptor.getOffsets());
    Value dst     = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGATHERB",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, offsets});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TLOG lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOLogToEmitC : public OpConversionPattern<pto::TLogOp> {
  using OpConversionPattern<pto::TLogOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TLogOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    ArrayAttr templateArgs;
    if (op.getPrecisionType() != pto::LogPrecision::Default) {
      StringRef precisionTok;
      switch (op.getPrecisionType()) {
      case pto::LogPrecision::Default:
        precisionTok = "pto::LogAlgorithm::DEFAULT";
        break;
      case pto::LogPrecision::HighPrecision:
        precisionTok = "pto::LogAlgorithm::HIGH_PRECISION";
        break;
      }
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, precisionTok)});
    }
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TLOG",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};



//===----------------------------------------------------------------------===//
// TLRELU lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

	struct PTOLReluToEmitC : public OpConversionPattern<pto::TLReluOp> {
	  using OpConversionPattern<pto::TLReluOp>::OpConversionPattern;
	
	  LogicalResult matchAndRewrite(pto::TLReluOp op, OpAdaptor adaptor,
	                                ConversionPatternRewriter &rewriter) const override {
	    auto loc = op.getLoc();
	
	    Value src = peelUnrealized(adaptor.getSrc());
	    Value slope = peelUnrealized(adaptor.getSlope());
	    Value dst = peelUnrealized(adaptor.getDst());

            SmallVector<Value, 3> operands{dst, src, slope};

	    rewriter.create<emitc::CallOpaqueOp>(
	        loc, TypeRange{}, "TLRELU",
	        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
	        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TMAX lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOMaxToEmitC : public OpConversionPattern<pto::TMaxOp> {
  using OpConversionPattern<pto::TMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TMAX", operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TMAXS lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

	struct PTOMaxSToEmitC : public OpConversionPattern<pto::TMaxSOp> {
	  using OpConversionPattern<pto::TMaxSOp>::OpConversionPattern;
	
	  LogicalResult matchAndRewrite(pto::TMaxSOp op, OpAdaptor adaptor,
	                                ConversionPatternRewriter &rewriter) const override {
	    Value src0 = peelUnrealized(adaptor.getSrc());
	    Value scalar = peelUnrealized(adaptor.getScalar());
	    Value dst  = peelUnrealized(adaptor.getDst());

	    SmallVector<Value, 3> operands{dst, src0, scalar};
	    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
	                                 "TMAXS", operands);

    rewriter.eraseOp(op);
    return success();
  }
};


//===----------------------------------------------------------------------===//
// TMIN lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOMinToEmitC : public OpConversionPattern<pto::TMinOp> {
  using OpConversionPattern<pto::TMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TMIN", operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TMINS lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TMINS lowering to EmitC (fix APFloat -> FloatAttr)  (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOMinsToEmitC : public OpConversionPattern<pto::TMinSOp> {
  using OpConversionPattern<pto::TMinSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMinSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    SmallVector<Value, 3> operands{dst, src, scalar};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TMINS", operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering for TMOV op -> EmitC)
//===----------------------------------------------------------------------===//

struct PTOMovToEmitC : public OpConversionPattern<pto::TMovOp> {
  using OpConversionPattern<pto::TMovOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMovOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value fp;
    if (op.getFp())
      fp = peelUnrealized(adaptor.getFp());
    Value preQuantScalar;
    if (op.getPreQuantScalar())
      preQuantScalar = peelUnrealized(adaptor.getPreQuantScalar());

    auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
    auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
    if (!dstOT || !srcOT)
      return rewriter.notifyMatchFailure(
          op, "tmov lowering expects opaque dst/src types");

    auto modeTok = [&](pto::AccToVecMode mode) -> StringRef {
      switch (mode) {
      case pto::AccToVecMode::SingleModeVec0:
        return "pto::AccToVecMode::SingleModeVec0";
      case pto::AccToVecMode::SingleModeVec1:
        return "pto::AccToVecMode::SingleModeVec1";
      case pto::AccToVecMode::DualModeSplitM:
        return "pto::AccToVecMode::DualModeSplitM";
      case pto::AccToVecMode::DualModeSplitN:
        return "pto::AccToVecMode::DualModeSplitN";
      }
      llvm_unreachable("unknown AccToVecMode");
    };

    auto modeAttr = op.getAccToVecModeAttr();
    auto reluTok = [&](pto::ReluPreMode mode) -> StringRef {
      switch (mode) {
      case pto::ReluPreMode::NoRelu:
        return "ReluPreMode::NoRelu";
      case pto::ReluPreMode::NormalRelu:
        return "ReluPreMode::NormalRelu";
      case pto::ReluPreMode::ScalarRelu:
        return "ReluPreMode::ScalarRelu";
      case pto::ReluPreMode::VectorRelu:
        return "ReluPreMode::VectorRelu";
      case pto::ReluPreMode::Pwl:
        return "ReluPreMode::Pwl";
      }
      llvm_unreachable("unknown ReluPreMode");
    };

    const bool hasFp = static_cast<bool>(fp);
    const bool hasPreQuantScalar = static_cast<bool>(preQuantScalar);
    const bool hasMode = static_cast<bool>(modeAttr);
    const bool reluNonDefault = op.getReluPreMode() != pto::ReluPreMode::NoRelu;

    SmallVector<Value, 4> operands{dst, src};
    SmallVector<Attribute, 5> templateArgVec{
        emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
        emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
    };
    StringRef callee = "TMOV";

    if (hasFp) {
      auto fpOT = mlir::dyn_cast<emitc::OpaqueType>(fp.getType());
      if (!fpOT)
        return rewriter.notifyMatchFailure(
            op, "tmov fp lowering expects opaque fp type");
      operands.push_back(fp);
      templateArgVec.push_back(emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()));
      if (hasMode)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, modeTok(modeAttr.getValue())));
      if (hasMode || reluNonDefault)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, reluTok(op.getReluPreMode())));
      callee = hasMode ? "TMOV" : "TMOV_FP";
    } else if (hasPreQuantScalar) {
      operands.push_back(preQuantScalar);
      if (hasMode)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, modeTok(modeAttr.getValue())));
      if (hasMode || reluNonDefault)
        templateArgVec.push_back(
            emitc::OpaqueAttr::get(ctx, reluTok(op.getReluPreMode())));
    } else if (hasMode) {
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, modeTok(modeAttr.getValue())));
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, reluTok(op.getReluPreMode())));
    } else if (reluNonDefault) {
      templateArgVec.push_back(
          emitc::OpaqueAttr::get(ctx, reluTok(op.getReluPreMode())));
    }

    ArrayAttr templateArgs =
        templateArgVec.size() == 2 && !hasFp && !hasPreQuantScalar &&
                !hasMode && !reluNonDefault
            ? ArrayAttr{}
            : rewriter.getArrayAttr(templateArgVec);

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, callee,
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMOV_FP DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMovFPToEmitC : public OpConversionPattern<pto::TMovFPOp> {
  using OpConversionPattern<pto::TMovFPOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMovFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());
    Value fp  = peelUnrealized(adaptor.getFp());

    // TMOV_FP<DstTileData, AccTile, FbTile>(dstTileData, cTile, fbTile)
    ArrayAttr templateArgs;
    auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
    auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
    auto fpOT  = mlir::dyn_cast<emitc::OpaqueType>(fp.getType());
    if (dstOT && srcOT && fpOT) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    SmallVector<Value, 3> operands{dst, src, fp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMOV_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOQuantToEmitC : public OpConversionPattern<pto::TQuantOp> {
  using OpConversionPattern<pto::TQuantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TQuantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());
    Value fp  = peelUnrealized(adaptor.getFp());

    // Optional offset (INT8_ASYM only): passed as pointer (&offset)
    Value offsetPtr;
    if (op.getOffset()) {
      Value offset = peelUnrealized(adaptor.getOffset());
      auto offsetOT = mlir::dyn_cast<emitc::OpaqueType>(offset.getType());
      if (offsetOT) {
        offsetPtr = rewriter
                        .create<emitc::ApplyOp>(
                            loc, emitc::PointerType::get(offsetOT), "&", offset)
                        .getResult();
      }
    }

    // TQUANT<QuantType, DstTile, SrcTile, FpTile>(dst, src, fp[, &offset])
    std::string quantTypeStr =
        op.getQuantType() == pto::QuantType::INT8_SYM
            ? "pto::QuantType::INT8_SYM"
            : "pto::QuantType::INT8_ASYM";
    ArrayAttr templateArgs;
    auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
    auto srcOT = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
    auto fpOT  = mlir::dyn_cast<emitc::OpaqueType>(fp.getType());
    if (dstOT && srcOT && fpOT) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, quantTypeStr),
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    SmallVector<Value> operands{dst, src, fp};
    if (offsetPtr)
      operands.push_back(offsetPtr);

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TQUANT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTODequantToEmitC : public OpConversionPattern<pto::TDequantOp> {
  using OpConversionPattern<pto::TDequantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDequantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scale  = peelUnrealized(adaptor.getScale());
    Value offset = peelUnrealized(adaptor.getOffset());

    // TDEQUANT<DstTile, SrcTile, ParaTile>(dst, src, scale, offset)
    ArrayAttr templateArgs;
    auto dstOT   = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
    auto srcOT   = mlir::dyn_cast<emitc::OpaqueType>(src.getType());
    auto scaleOT = mlir::dyn_cast<emitc::OpaqueType>(scale.getType());
    if (dstOT && srcOT && scaleOT) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, scaleOT.getValue().str()),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TDEQUANT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/SmallVector<Value>{dst, src, scale, offset});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMRGSORT DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMrgSortToEmitC : public OpConversionPattern<pto::TMrgSortOp> {
  using OpConversionPattern<pto::TMrgSortOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMrgSortOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    if (op.isFormat1()) {
      Value src = peelUnrealized(adaptor.getSrcs().front());
      Value dst = peelUnrealized(adaptor.getDsts().front());
      Value blockLen = peelUnrealized(adaptor.getBlockLen());

      SmallVector<Value, 3> operands{dst, src, blockLen};
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TMRGSORT",
          ArrayAttr{}, ArrayAttr{}, operands);
    } else if (op.isFormat2()) {
      // pto-isa API:
      //   TMRGSORT<DstTile, TmpTile, Src0, Src1[, Src2[, Src3]], exhausted>(
      //       dst, executedNumList, tmp, src0, src1[, src2[, src3]]);
      auto *ctx = rewriter.getContext();

      Value dst = peelUnrealized(adaptor.getDsts()[0]);
      Value tmp = peelUnrealized(adaptor.getTmp());
      Value excuted = peelUnrealized(adaptor.getExcuted());

      SmallVector<Value, 4> srcs;
      srcs.reserve(adaptor.getSrcs().size());
      for (Value v : adaptor.getSrcs())
        srcs.push_back(peelUnrealized(v));

      auto dstOT = mlir::dyn_cast<emitc::OpaqueType>(dst.getType());
      auto tmpOT = mlir::dyn_cast<emitc::OpaqueType>(tmp.getType());
      if (!dstOT || !tmpOT || srcs.size() < 2 || srcs.size() > 4)
        return op.emitOpError("format2 expects dst/tmp tilebufs and 2 to 4 srcs");

      SmallVector<Attribute, 8> targs;
      targs.reserve(2 + srcs.size() + 1);
      targs.push_back(emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()));
      targs.push_back(emitc::OpaqueAttr::get(ctx, tmpOT.getValue().str()));
      for (Value v : srcs) {
        auto ot = mlir::dyn_cast<emitc::OpaqueType>(v.getType());
        if (!ot)
          return op.emitOpError("format2 expects tilebuf srcs");
        targs.push_back(emitc::OpaqueAttr::get(ctx, ot.getValue().str()));
      }
      targs.push_back(emitc::OpaqueAttr::get(ctx, op.getExhausted() ? "true" : "false"));
      ArrayAttr templateArgs = rewriter.getArrayAttr(targs);

      SmallVector<Value, 7> operands{dst, excuted, tmp};
      operands.append(srcs.begin(), srcs.end());

      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TMRGSORT",
          /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs, operands);
    } else {
      return op.emitOpError("unsupported mrgsort_dps format");
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMUL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMulToEmitC : public OpConversionPattern<pto::TMulOp> {
  using OpConversionPattern<pto::TMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TMUL", operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMULS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMulsToEmitC : public OpConversionPattern<pto::TMulSOp> {
  using OpConversionPattern<pto::TMulSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMulSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc0());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    SmallVector<Value, 3> operands{dst, src, scalar};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TMULS", operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TNEG DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTONegToEmitC : public OpConversionPattern<pto::TNegOp> {
  using OpConversionPattern<pto::TNegOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TNegOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TNEG",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TNOT DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTONotToEmitC : public OpConversionPattern<pto::TNotOp> {
  using OpConversionPattern<pto::TNotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TNotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TNOT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TOR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOOrToEmitC : public OpConversionPattern<pto::TOrOp> {
  using OpConversionPattern<pto::TOrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TOrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TOR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TORS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOOrsToEmitC : public OpConversionPattern<pto::TOrSOp> {
  using OpConversionPattern<pto::TOrSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TOrSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc());
    Value dst  = peelUnrealized(adaptor.getDst());
    // NOTE: The conversion type system may materialize integers as emitc.opaque
    // (e.g. "int32_t"). For EmitC call emission we can pass the scalar through
    // directly without arith casts here.
    Value s = adaptor.getScalar();

    SmallVector<Value, 3> operands{dst, src0, s};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TORS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTADD DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartAddToEmitC : public OpConversionPattern<pto::TPartAddOp> {
  using OpConversionPattern<pto::TPartAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTADD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMAX DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartMaxToEmitC : public OpConversionPattern<pto::TPartMaxOp> {
  using OpConversionPattern<pto::TPartMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMIN DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartMinToEmitC : public OpConversionPattern<pto::TPartMinOp> {
  using OpConversionPattern<pto::TPartMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOPartArgMaxToEmitC
    : public OpConversionPattern<pto::TPartArgMaxOp> {
  using OpConversionPattern<pto::TPartArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src0Idx = peelUnrealized(adaptor.getSrc0Idx());
    Value src1Idx = peelUnrealized(adaptor.getSrc1Idx());
    Value dst = peelUnrealized(adaptor.getDst());
    Value dstIdx = peelUnrealized(adaptor.getDstIdx());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TPARTARGMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1, dstIdx, src0Idx, src1Idx});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOPartArgMinToEmitC
    : public OpConversionPattern<pto::TPartArgMinOp> {
  using OpConversionPattern<pto::TPartArgMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartArgMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src0Idx = peelUnrealized(adaptor.getSrc0Idx());
    Value src1Idx = peelUnrealized(adaptor.getSrc1Idx());
    Value dst = peelUnrealized(adaptor.getDst());
    Value dstIdx = peelUnrealized(adaptor.getDstIdx());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TPARTARGMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1, dstIdx, src0Idx, src1Idx});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMUL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartMulToEmitC : public OpConversionPattern<pto::TPartMulOp> {
  using OpConversionPattern<pto::TPartMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPartMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMUL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPRELU DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPreluToEmitC : public OpConversionPattern<pto::TPReluOp> {
  using OpConversionPattern<pto::TPReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst  = peelUnrealized(adaptor.getDst());

    // C++ interface: TPRELU(dst, src0, src1, tmp) — last parameter is tmp.
    SmallVector<Value, 4> operands{dst, src0, src1, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPRELU",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRECIP DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORecipToEmitC : public OpConversionPattern<pto::TRecipOp> {
  using OpConversionPattern<pto::TRecipOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRecipOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    ArrayAttr templateArgs;
    if (op.getPrecisionType() != pto::RecipPrecision::Default) {
      StringRef precisionTok;
      switch (op.getPrecisionType()) {
      case pto::RecipPrecision::Default:
        precisionTok = "pto::RecipAlgorithm::DEFAULT";
        break;
      case pto::RecipPrecision::HighPrecision:
        precisionTok = "pto::RecipAlgorithm::HIGH_PRECISION";
        break;
      }
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, precisionTok)});
    }
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRECIP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRELU DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOReluToEmitC : public OpConversionPattern<pto::TReluOp> {
  using OpConversionPattern<pto::TReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRELU",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TREM DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORemToEmitC : public OpConversionPattern<pto::TRemOp> {
  using OpConversionPattern<pto::TRemOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRemOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst  = peelUnrealized(adaptor.getDst());
    SmallVector<Value, 4> operands{dst, src0, src1, tmp};
    ArrayAttr templateArgs;
    if (op.getPrecisionType() != pto::RemPrecision::Default) {
      StringRef precisionTok;
      switch (op.getPrecisionType()) {
      case pto::RemPrecision::Default:
        precisionTok = "pto::RemAlgorithm::DEFAULT";
        break;
      case pto::RemPrecision::HighPrecision:
        precisionTok = "pto::RemAlgorithm::HIGH_PRECISION";
        break;
      }
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, precisionTok)});
    }
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TREM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOFModToEmitC : public OpConversionPattern<pto::TFModOp> {
  using OpConversionPattern<pto::TFModOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFModOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    ArrayAttr templateArgs;
    if (op.getPrecisionType() != pto::FmodPrecision::Default) {
      StringRef precisionTok;
      switch (op.getPrecisionType()) {
      case pto::FmodPrecision::Default:
        precisionTok = "pto::FmodAlgorithm::DEFAULT";
        break;
      case pto::FmodPrecision::HighPrecision:
        precisionTok = "pto::FmodAlgorithm::HIGH_PRECISION";
        break;
      }
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, precisionTok)});
    }
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFMOD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TREMS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORemSToEmitC : public OpConversionPattern<pto::TRemSOp> {
  using OpConversionPattern<pto::TRemSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRemSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());
    SmallVector<Value, 4> operands{dst, src, scalar, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TREMS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOFModSToEmitC : public OpConversionPattern<pto::TFModSOp> {
  using OpConversionPattern<pto::TFModSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TFModSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFMODS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPAND DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowExpandToEmitC : public OpConversionPattern<pto::TRowExpandOp> {
  using OpConversionPattern<pto::TRowExpandOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPAND",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowExpandAddToEmitC : public OpConversionPattern<pto::TRowExpandAddOp> {
  using OpConversionPattern<pto::TRowExpandAddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandAddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDADD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowExpandExpdifToEmitC
    : public OpConversionPattern<pto::TRowExpandExpdifOp> {
  using OpConversionPattern<pto::TRowExpandExpdifOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandExpdifOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());
    Value tmp  = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVector<Value, 4> operands;
    if (tmp)
      operands.assign({dst, src0, src1, tmp});
    else
      operands.assign({dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDEXPDIF",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

static void replaceOrEraseWithOpaqueCallAndReturnDst(Operation *op, Value dst,
                                                     StringRef callee,
                                                     ArrayRef<Value> args,
                                                     ArrayAttr templateArgs,
                                                     ConversionPatternRewriter &rewriter) {
  createLastUseAwareOpaqueCall(rewriter, op, TypeRange{}, callee, args, ArrayAttr{}, templateArgs);
  if (op->getNumResults() == 1)
    rewriter.replaceOp(op, dst);
  else
    rewriter.eraseOp(op);
}

// ---------- TOp ----------
struct PTOTGemvBiasToTGEMV_BIAS
    : public OpConversionPattern<pto::TGemvBiasOp> {
  using OpConversionPattern<pto::TGemvBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a    = peelUnrealized(adaptor.getA());
    Value b    = peelUnrealized(adaptor.getB());
    Value bias = peelUnrealized(adaptor.getBias());
    Value dst  = peelUnrealized(adaptor.getDst());

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_BIAS",
                                             {dst, a, b, bias}, templateArgs, rewriter);
    return success();
  }
};

struct PTOTGemvMXToTGEMV_MX
    : public OpConversionPattern<pto::TGemvMxOp> {
  using OpConversionPattern<pto::TGemvMxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvMxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_MX",
                                             {dst, a, aScale, b, bScale}, templateArgs,
                                             rewriter);
    return success();
  }
};

struct PTOTGemvMXAccToTGEMV_MX
    : public OpConversionPattern<pto::TGemvMxAccOp> {
  using OpConversionPattern<pto::TGemvMxAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvMxAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value cIn     = peelUnrealized(adaptor.getCIn());
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_MX",
                                             {dst, cIn, a, aScale, b, bScale}, templateArgs,
                                             rewriter);
    return success();
  }
};

struct PTOTGemvMXBiasToTGEMV_MX
    : public OpConversionPattern<pto::TGemvMxBiasOp> {
  using OpConversionPattern<pto::TGemvMxBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvMxBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value bias    = peelUnrealized(adaptor.getBias());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TGEMV_MX",
                                             {dst, a, aScale, b, bScale, bias}, ArrayAttr{},
                                             rewriter);
    return success();
  }
};

struct PTOTMatmulBiasToTMATMUL_BIAS
    : public OpConversionPattern<pto::TMatmulBiasOp> {
  using OpConversionPattern<pto::TMatmulBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a    = peelUnrealized(adaptor.getA());
    Value b    = peelUnrealized(adaptor.getB());
    Value bias = peelUnrealized(adaptor.getBias());
    Value dst  = peelUnrealized(adaptor.getDst());

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TMATMUL_BIAS",
                                             {dst, a, b, bias}, templateArgs, rewriter);
    return success();
  }
};

struct PTOTMatmulMXToTMATMUL_MX
    : public OpConversionPattern<pto::TMatmulMxOp> {
  using OpConversionPattern<pto::TMatmulMxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TMATMUL_MX",
                                             {dst, a, aScale, b, bScale}, templateArgs,
                                             rewriter);
    return success();
  }
};

struct PTOTMatmulMXAccToTMATMUL_MX_ACC
    : public OpConversionPattern<pto::TMatmulMxAccOp> {
  using OpConversionPattern<pto::TMatmulMxAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value cIn     = peelUnrealized(adaptor.getCIn());
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    ArrayAttr templateArgs =
        buildAccPhaseTemplateArgs(rewriter, op.getAccPhase());
    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TMATMUL_MX",
                                             {dst, cIn, a, aScale, b, bScale}, templateArgs,
                                             rewriter);
    return success();
  }
};

struct PTOTMatmulMXBiasToTMATMUL_MX_BIAS
    : public OpConversionPattern<pto::TMatmulMxBiasOp> {
  using OpConversionPattern<pto::TMatmulMxBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value bias    = peelUnrealized(adaptor.getBias());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCallAndReturnDst(op.getOperation(), dst, "TMATMUL_MX",
                                             {dst, a, aScale, b, bScale, bias}, ArrayAttr{},
                                             rewriter);
    return success();
  }
};

struct PTORowExpandDivToEmitC : public OpConversionPattern<pto::TRowExpandDivOp> {
  using OpConversionPattern<pto::TRowExpandDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandDivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());
    Value tmp  = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVector<Value, 4> operands;
    if (tmp)
      operands.assign({dst, src0, src1, tmp});
    else
      operands.assign({dst, src0, src1});
    ArrayAttr templateArgs;
    if (op.getPrecisionType() != pto::DivPrecision::Default) {
      StringRef precisionTok;
      switch (op.getPrecisionType()) {
      case pto::DivPrecision::Default:
        precisionTok = "pto::DivAlgorithm::DEFAULT";
        break;
      case pto::DivPrecision::HighPrecision:
        precisionTok = "pto::DivAlgorithm::HIGH_PRECISION";
        break;
      }
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, precisionTok)});
    }
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TROWEXPANDDIV", operands, ArrayAttr{}, templateArgs);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPANDMUL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowExpandMulToEmitC : public OpConversionPattern<pto::TRowExpandMulOp> {
  using OpConversionPattern<pto::TRowExpandMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());
    Value tmp  = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVector<Value, 4> operands;
    if (tmp)
      operands.assign({dst, src0, src1, tmp});
    else
      operands.assign({dst, src0, src1});
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TROWEXPANDMUL", operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPANDSUB DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowExpandSubToEmitC : public OpConversionPattern<pto::TRowExpandSubOp> {
  using OpConversionPattern<pto::TRowExpandSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());
    Value tmp  = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVector<Value, 4> operands;
    if (tmp)
      operands.assign({dst, src0, src1, tmp});
    else
      operands.assign({dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDSUB",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowExpandMaxToEmitC : public OpConversionPattern<pto::TRowExpandMaxOp> {
  using OpConversionPattern<pto::TRowExpandMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());
    Value tmp  = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVector<Value, 4> operands;
    if (tmp)
      operands.assign({dst, src0, src1, tmp});
    else
      operands.assign({dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowExpandMinToEmitC : public OpConversionPattern<pto::TRowExpandMinOp> {
  using OpConversionPattern<pto::TRowExpandMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowExpandMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());
    Value tmp  = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVector<Value, 4> operands;
    if (tmp)
      operands.assign({dst, src0, src1, tmp});
    else
      operands.assign({dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWMAX DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowMaxToEmitC : public OpConversionPattern<pto::TRowMaxOp> {
  using OpConversionPattern<pto::TRowMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src, tmp};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TROWMAX", operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowArgMaxToEmitC
    : public OpConversionPattern<pto::TRowArgMaxOp> {
  using OpConversionPattern<pto::TRowArgMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowArgMaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWARGMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWMIN DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowMinToEmitC : public OpConversionPattern<pto::TRowMinOp> {
  using OpConversionPattern<pto::TRowMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src, tmp};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TROWMIN", operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowArgMinToEmitC
    : public OpConversionPattern<pto::TRowArgMinOp> {
  using OpConversionPattern<pto::TRowArgMinOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowArgMinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWARGMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, tmp});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWSUM DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowSumToEmitC : public OpConversionPattern<pto::TRowSumOp> {
  using OpConversionPattern<pto::TRowSumOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowSumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src, tmp};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TROWSUM", operands);

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTORowProdToEmitC : public OpConversionPattern<pto::TRowProdOp> {
  using OpConversionPattern<pto::TRowProdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRowProdOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWPROD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRSQRT DPS/memref op)
// - no-tmp form : TRSQRT(dst, src)
// - tmp form    : TRSQRT(dst, src, tmp)
//===----------------------------------------------------------------------===//

struct PTORsqrtToEmitC : public OpConversionPattern<pto::TRsqrtOp> {
  using OpConversionPattern<pto::TRsqrtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TRsqrtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    SmallVector<Value, 3> operands{dst, src};
    if (Value tmp = adaptor.getTmp())
      operands.push_back(peelUnrealized(tmp));
    ArrayAttr templateArgs;
    if (op.getPrecisionType() != pto::RsqrtPrecision::Default) {
      StringRef precisionTok;
      switch (op.getPrecisionType()) {
      case pto::RsqrtPrecision::Default:
        precisionTok = "pto::RsqrtAlgorithm::DEFAULT";
        break;
      case pto::RsqrtPrecision::HighPrecision:
        precisionTok = "pto::RsqrtAlgorithm::HIGH_PRECISION";
        break;
      }
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, precisionTok)});
    }
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRSQRT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSCATTER DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOScatterToEmitC : public OpConversionPattern<pto::TScatterOp> {
  using OpConversionPattern<pto::TScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TScatterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const bool hasMaskPattern = static_cast<bool>(op.getMaskPatternAttr());
    const bool hasIndexes = static_cast<bool>(op.getIndexes());
    if (hasMaskPattern == hasIndexes) {
      return rewriter.notifyMatchFailure(
          op, "expected exactly one of indexes operand or maskPattern attribute");
    }

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    if (auto mp = op.getMaskPatternAttr()) {
      auto *ctx = rewriter.getContext();
      auto targs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, maskPatternTok(mp)),
      });
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TSCATTER",
          /*args=*/ArrayAttr{}, /*templateArgs=*/targs,
          /*operands=*/ValueRange{dst, src});
    } else {
      Value idx = peelUnrealized(adaptor.getIndexes());
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TSCATTER",
          /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
          /*operands=*/ValueRange{dst, src, idx});
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSEL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSelToEmitC : public OpConversionPattern<pto::TSelOp> {
  using OpConversionPattern<pto::TSelOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value mask = peelUnrealized(adaptor.getMask());
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 5> operands{dst, mask, src0, src1, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSEL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSELS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSelSToEmitC : public OpConversionPattern<pto::TSelSOp> {
  using OpConversionPattern<pto::TSelSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSelSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value mask = peelUnrealized(adaptor.getMask());
    Value src  = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 5> operands{dst, mask, src, tmp, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSELS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSHL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOShlSToEmitC : public OpConversionPattern<pto::TShlOp> {
  using OpConversionPattern<pto::TShlOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShlOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSHR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOShrSToEmitC : public OpConversionPattern<pto::TShrOp> {
  using OpConversionPattern<pto::TShrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering for TSHLS/TSHRS DPS: shift by scalar)
//===----------------------------------------------------------------------===//

struct PTOShlSConstToEmitC : public OpConversionPattern<pto::TShlSOp> {
  using OpConversionPattern<pto::TShlSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShlSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHLS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOShrSConstToEmitC : public OpConversionPattern<pto::TShrSOp> {
  using OpConversionPattern<pto::TShrSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TShrSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHRS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (TSORT32 DPS/memref op: ins(src, idx[, tmp]) outs(dst))
//===----------------------------------------------------------------------===//

struct PTOSORT32SToEmitC : public OpConversionPattern<pto::TSort32Op> {
  using OpConversionPattern<pto::TSort32Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSort32Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value idx = peelUnrealized(adaptor.getIdx());
    Value tmp = op.getTmp() ? peelUnrealized(adaptor.getTmp()) : Value();

    SmallVector<Value, 4> operands;
    if (tmp)
      operands.assign({dst, src, idx, tmp});
    else
      operands.assign({dst, src, idx});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSORT32",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSQRT DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSqrtSToEmitC : public OpConversionPattern<pto::TSqrtOp> {
  using OpConversionPattern<pto::TSqrtOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSqrtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src};
    ArrayAttr templateArgs;
    if (op.getPrecisionType() != pto::SqrtPrecision::Default) {
      StringRef precisionTok;
      switch (op.getPrecisionType()) {
      case pto::SqrtPrecision::Default:
        precisionTok = "pto::SqrtAlgorithm::DEFAULT";
        break;
      case pto::SqrtPrecision::HighPrecision:
        precisionTok = "pto::SqrtAlgorithm::HIGH_PRECISION";
        break;
      }
      templateArgs = rewriter.getArrayAttr(
          {emitc::OpaqueAttr::get(ctx, precisionTok)});
    }
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSQRT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSTORE_FP DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOStoreFPSToEmitC : public OpConversionPattern<pto::TStoreFPOp> {
  using OpConversionPattern<pto::TStoreFPOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TStoreFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value fp = peelUnrealized(adaptor.getFp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src, fp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSTORE_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUB DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSToEmitC : public OpConversionPattern<pto::TSubOp> {
  using OpConversionPattern<pto::TSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src0, src1};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TSUB", operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBC DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubCSToEmitC : public OpConversionPattern<pto::TSubCOp> {
  using OpConversionPattern<pto::TSubCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src2 = peelUnrealized(adaptor.getSrc2());
    Value dst = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TSUBC yet.
    // Decompose: dst = src0 - src1 + src2
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSUB",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, dst, src2});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSSToEmitC : public OpConversionPattern<pto::TSubSOp> {
  using OpConversionPattern<pto::TSubSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src, scalar};
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TSUBS", operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBSC DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSCToEmitC : public OpConversionPattern<pto::TSubSCOp> {
  using OpConversionPattern<pto::TSubSCOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TSubSCOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TSUBSC yet.
    // Decompose: dst = src0 - scalar + src1
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSUBS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, scalar});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, dst, src1});

    rewriter.eraseOp(op);
    return success();
  }
};


//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TXOR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOXORToEmitC : public OpConversionPattern<pto::TXorOp> {
  using OpConversionPattern<pto::TXorOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TXorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());
    Value tmp = peelUnrealized(adaptor.getTmp());
    SmallVector<Value, 4> operands{dst, src0, src1, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TXOR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOTTransToEmitC : public OpConversionPattern<pto::TTransOp> {
  using OpConversionPattern<pto::TTransOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TTransOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TTRANS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TXORS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOXORSToEmitC : public OpConversionPattern<pto::TXorSOp> {
  using OpConversionPattern<pto::TXorSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TXorSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value tmp  = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src, scalar, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TXORS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
  struct PTOPrintToTPRINT : public OpConversionPattern<pto::TPrintOp> {
  using OpConversionPattern<pto::TPrintOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TPrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());

    SmallVector<Value, 4> operands{src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPRINT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

// pto.print "format", %scalar -> PRINTF("format", scalar)
struct PTOPrintOpToEmitC : public OpConversionPattern<pto::PrintOp> {
  using OpConversionPattern<pto::PrintOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PrintOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    std::string fmt = op.getFormat().str();
    if (fmt.empty())
      fmt = "%f";
    std::string quoted = "\"";
    for (char c : fmt) {
      if (c == '"' || c == '\\')
        quoted += '\\';
      else if (c == '\n')
        quoted += "\\n";
      else if (c == '\t')
        quoted += "\\t";
      else
        quoted += c;
    }
    quoted += "\"";

    Value scalar = peelUnrealized(adaptor.getScalar());
    auto argsAttr = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(ctx, quoted),
         IntegerAttr::get(IndexType::get(ctx), 0)});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "cce::printf",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{scalar});

    rewriter.eraseOp(op);
    return success();
  }
};

// pto.trap -> TRAP()
struct PTOTrapOpToEmitC : public OpConversionPattern<pto::TrapOp> {
  using OpConversionPattern<pto::TrapOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TrapOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "trap",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});

    rewriter.eraseOp(op);
    return success();
  }
};

// =============================================================================
// 2. BindTileOp Lowering (FIX: Trace back to physical address)
// =============================================================================
struct PTOBindTileToEmitC : public OpConversionPattern<pto::BindTileOp> {
  using OpConversionPattern::OpConversionPattern;

  struct TileBuildSpec {
    std::string tileTypeStr;
    bool useConstructor = false;
    SmallVector<Value> constructorArgs;
  };

  static bool getIndexConst(Value v, int64_t &out) {
    if (!v)
      return false;
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue())) {
        out = ia.getValue().getSExtValue();
        return true;
      }
    }
    return false;
  }

  static bool getTilePointerStrides(pto::TileBufConfigAttr configAttr,
                                    Type elemTy, int64_t rows, int64_t cols,
                                    int64_t &rowStride,
                                    int64_t &colStride) {
    if (rows == ShapedType::kDynamic || cols == ShapedType::kDynamic)
      return false;

    int32_t blVal = 0;
    if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout()))
      blVal = static_cast<int32_t>(blAttr.getValue());
    else if (auto intAttr = dyn_cast<IntegerAttr>(configAttr.getBLayout()))
      blVal = static_cast<int32_t>(intAttr.getInt());

    int32_t slVal = 0;
    if (auto slAttr = dyn_cast<SLayoutAttr>(configAttr.getSLayout()))
      slVal = static_cast<int32_t>(slAttr.getValue());
    else if (auto intAttr = dyn_cast<IntegerAttr>(configAttr.getSLayout()))
      slVal = static_cast<int32_t>(intAttr.getInt());

    bool boxed = slVal != 0;
    int64_t innerRows = 1;
    int64_t innerCols = 1;
    if (boxed) {
      int32_t fractal = 512;
      if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
        fractal = static_cast<int32_t>(frAttr.getInt());

      unsigned elemBytes = pto::getPTOStorageElemByteSize(elemTy);
      if (elemBytes == 0)
        return false;

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
        if (slVal == 1) {
          innerRows = 16;
          innerCols = 32 / elemBytes;
        } else if (slVal == 2) {
          innerRows = 32 / elemBytes;
          innerCols = 16;
        } else {
          return false;
        }
        break;
      default:
        return false;
      }
      if (innerRows <= 0 || innerCols <= 0)
        return false;
    }

    if (!boxed) {
      if (blVal == 1) {
        rowStride = 1;
        colStride = rows;
      } else {
        rowStride = cols;
        colStride = 1;
      }
      return true;
    }

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

  LogicalResult matchAndRewrite(pto::BindTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto configAttr = op.getConfigAttr();
    auto viewSemantics = op->getAttrOfType<StringAttr>("pto.view_semantics");
    bool isSubView = viewSemantics && viewSemantics.getValue() == "subview";

    auto peelAllCasts = [](Value v) {
      while (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
        v = castOp.getOperand(0);
      if (auto castOp = v.getDefiningOp<emitc::CastOp>())
        v = castOp.getOperand();
      return v;
    };
    auto isTileLike = [](Value v) -> bool {
      auto ot = dyn_cast<emitc::OpaqueType>(v.getType());
      if (!ot)
        return false;
      StringRef s = ot.getValue();
      return s.contains("Tile<") || s.contains("ConvTile<");
    };
    auto buildTileSpec = [&]() -> FailureOr<TileBuildSpec> {
      auto resMrTy = dyn_cast<MemRefType>(op.getType());
      if (!resMrTy)
        return failure();

      const char *roleTok = "TileType::Vec";
      if (auto asAttr =
              dyn_cast_or_null<pto::AddressSpaceAttr>(resMrTy.getMemorySpace())) {
        switch (asAttr.getAddressSpace()) {
        case pto::AddressSpace::VEC:
          roleTok = "TileType::Vec";
          break;
        case pto::AddressSpace::MAT:
          roleTok = "TileType::Mat";
          break;
        case pto::AddressSpace::LEFT:
          roleTok = "TileType::Left";
          break;
        case pto::AddressSpace::RIGHT:
          roleTok = "TileType::Right";
          break;
        case pto::AddressSpace::ACC:
          roleTok = "TileType::Acc";
          break;
        case pto::AddressSpace::BIAS:
          roleTok = "TileType::Bias";
          break;
        case pto::AddressSpace::SCALING:
          roleTok = "TileType::Scaling";
          break;
        case pto::AddressSpace::GM:
        case pto::AddressSpace::Zero:
          roleTok = "TileType::Vec";
          break;
        }
      }

      Type elemTy = resMrTy.getElementType();
      Type emitElemTy = getTypeConverter()->convertType(elemTy);
      if (!emitElemTy)
        return failure();
      auto emitElemOpaque = dyn_cast<emitc::OpaqueType>(emitElemTy);
      if (!emitElemOpaque)
        return failure();
      std::string elemTypeStr = emitElemOpaque.getValue().str();

      if (resMrTy.getRank() < 2)
        return failure();
      int64_t rows = resMrTy.getDimSize(0);
      int64_t cols = resMrTy.getDimSize(1);
      if (rows == ShapedType::kDynamic || cols == ShapedType::kDynamic)
        return failure();

      std::string blTok = "BLayout::RowMajor";
      if (auto blAttr = dyn_cast<BLayoutAttr>(configAttr.getBLayout())) {
        if (static_cast<int32_t>(blAttr.getValue()) == 1)
          blTok = "BLayout::ColMajor";
      }
      pto::BLayout blayout = getTileBufBLayoutValue(configAttr);

      if (isSubView) {
        auto subMrTy = dyn_cast<MemRefType>(op.getSource().getType());
        auto subViewOp = op.getSource().getDefiningOp<memref::SubViewOp>();
        if (subMrTy && subMrTy.getRank() >= 2 && subViewOp) {
          int64_t subRows = subMrTy.getDimSize(0);
          int64_t subCols = subMrTy.getDimSize(1);
          SmallVector<int64_t> inheritedStrides;
          int64_t inheritedOffset = ShapedType::kDynamic;

          if (!pto::isPTOFloat4PackedType(elemTy) &&
              subRows != ShapedType::kDynamic &&
              subCols != ShapedType::kDynamic &&
              succeeded(getStridesAndOffset(subMrTy, inheritedStrides,
                                            inheritedOffset)) &&
              inheritedStrides.size() >= 2) {
            int64_t childRowStride = 0;
            int64_t childColStride = 0;
            bool sameStrides = getTilePointerStrides(
                configAttr, elemTy, subRows, subCols, childRowStride,
                childColStride);
            sameStrides = sameStrides &&
                          inheritedStrides[0] == childRowStride &&
                          inheritedStrides[1] == childColStride;
            if (sameStrides) {
              rows = subRows;
              cols = subCols;
            }
          }
        }
      }

      std::string slTok = "SLayout::NoneBox";
      if (auto slAttr = dyn_cast<SLayoutAttr>(configAttr.getSLayout())) {
        int32_t slVal = static_cast<int32_t>(slAttr.getValue());
        slTok = (slVal == 1) ? "SLayout::RowMajor"
                             : (slVal == 2) ? "SLayout::ColMajor"
                                            : "SLayout::NoneBox";
      }

      int32_t fractal = 512;
      if (auto frAttr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
        fractal = frAttr.getInt();

      std::string padTok = "PadValue::Null";
      if (auto padAttr = dyn_cast<PadValueAttr>(configAttr.getPad())) {
        switch (static_cast<int32_t>(padAttr.getValue())) {
        case 1:
          padTok = "PadValue::Zero";
          break;
        case 2:
          padTok = "PadValue::Max";
          break;
        case 3:
          padTok = "PadValue::Min";
          break;
        default:
          padTok = "PadValue::Null";
          break;
        }
      }

      std::string compactTok = "CompactMode::Null";
      if (auto compactAttr = dyn_cast<CompactModeAttr>(configAttr.getCompactMode())) {
        switch (static_cast<int32_t>(compactAttr.getValue())) {
        case 1:
          compactTok = "CompactMode::Normal";
          break;
        case 2:
          compactTok = "CompactMode::RowPlusOne";
          break;
        default:
          compactTok = "CompactMode::Null";
          break;
        }
      }

      std::string vrowTok, vcolTok;
      bool useConstructor = false;
      bool rowIsDynamic = false;
      bool colIsDynamic = false;
      SmallVector<Value> constructorArgs;

      Value vRow = op.getValidRow();
      Value vCol = op.getValidCol();
      Value vRowEmitC = adaptor.getValidRow();
      Value vColEmitC = adaptor.getValidCol();
      bool forceDynamicValid = op->hasAttr(kForceDynamicValidShapeAttrName);
      int64_t cRow = 0, cCol = 0;
      bool rowIsConst = vRow && getIndexConst(vRow, cRow);
      bool colIsConst = vCol && getIndexConst(vCol, cCol);

      auto makeCtorDimValue = [&](Value emitted, int64_t fallback) -> Value {
        if (emitted)
          return emitted;
        return makeEmitCIntConstant(
            rewriter, loc, emitc::OpaqueType::get(ctx, "int32_t"), fallback);
      };
      auto maybeScaleDynamicValid = [&](Value emitted, int dimIdx) -> Value {
        if (!emitted || !pto::isPTOFloat4PackedType(elemTy))
          return emitted;
        int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
        if (dimIdx != packedDim)
          return emitted;
        auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
        Value two = makeEmitCIntConstant(rewriter, loc, i32Ty, 2);
        return rewriter.create<emitc::MulOp>(loc, i32Ty, emitted, two).getResult();
      };

      if (forceDynamicValid) {
        vrowTok = "-1";
        vcolTok = "-1";
        useConstructor = true;
        constructorArgs.push_back(
            makeCtorDimValue(maybeScaleDynamicValid(vRowEmitC, 0),
                             renderTileTemplateDim(rowIsConst ? cRow : rows,
                                                   elemTy, blayout, 0)));
        constructorArgs.push_back(
            makeCtorDimValue(maybeScaleDynamicValid(vColEmitC, 1),
                             renderTileTemplateDim(colIsConst ? cCol : cols,
                                                   elemTy, blayout, 1)));
      } else {
        if (rowIsConst) {
          vrowTok = std::to_string(
              renderTileTemplateDim(cRow, elemTy, blayout, 0));
        } else if (vRow) {
          vrowTok = "-1";
          rowIsDynamic = true;
          useConstructor = true;
        } else {
          vrowTok = std::to_string(
              renderTileTemplateDim(rows, elemTy, blayout, 0));
        }

        if (colIsConst) {
          vcolTok = std::to_string(
              renderTileTemplateDim(cCol, elemTy, blayout, 1));
        } else if (vCol) {
          vcolTok = "-1";
          colIsDynamic = true;
          useConstructor = true;
        } else {
          vcolTok = std::to_string(
              renderTileTemplateDim(cols, elemTy, blayout, 1));
        }

        if (useConstructor) {
          if (rowIsDynamic && vRowEmitC)
            constructorArgs.push_back(maybeScaleDynamicValid(vRowEmitC, 0));
          if (colIsDynamic && vColEmitC)
            constructorArgs.push_back(maybeScaleDynamicValid(vColEmitC, 1));
        }
      }

      std::string tileTypeStr = std::string("Tile<") + roleTok + ", " +
                                elemTypeStr + ", " +
                                std::to_string(renderTileTemplateDim(
                                    rows, elemTy, blayout, 0)) +
                                ", " +
                                std::to_string(renderTileTemplateDim(
                                    cols, elemTy, blayout, 1)) +
                                ", " + blTok +
                                ", " + vrowTok + ", " + vcolTok + ", " + slTok +
                                ", " + std::to_string(fractal) + ", " + padTok +
                                ", " + compactTok +
                                ">";
      return TileBuildSpec{tileTypeStr, useConstructor, constructorArgs};
    };

    auto buildTileValue = [&](const TileBuildSpec &spec,
                              bool forceDeclaration = false) -> Value {
      auto tileType = emitc::OpaqueType::get(ctx, spec.tileTypeStr);
      if (spec.useConstructor && !forceDeclaration) {
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, tileType, spec.tileTypeStr,
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange(spec.constructorArgs))
            .getResult(0);
      }

      return rewriter
          .create<emitc::VariableOp>(loc, tileType, emitc::OpaqueAttr::get(ctx, ""))
          .getResult();
    };

    auto emitElemTypeToString = [&](Type elemTy) -> std::string {
      return getEmitCScalarTypeToken(elemTy);
    };

    auto buildIntegralAddress = [&](Value sourceValue) -> FailureOr<Value> {
      auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
      auto rcU64 =
          rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});

      Value rawPtr = sourceValue;
      if (auto ot = dyn_cast<emitc::OpaqueType>(sourceValue.getType())) {
        StringRef tyStr = ot.getValue();
        if (tyStr.contains("Tile<") || tyStr.contains("ConvTile<")) {
          auto srcMrTy = dyn_cast<MemRefType>(op.getSource().getType());
          if (!srcMrTy)
            return failure();
          std::string elemTok = emitElemTypeToString(srcMrTy.getElementType());
          pto::AddressSpace as = pto::AddressSpace::GM;
          if (auto asAttr =
                  dyn_cast_or_null<pto::AddressSpaceAttr>(srcMrTy.getMemorySpace()))
            as = asAttr.getAddressSpace();
          rawPtr = materializeTileDataValue(rewriter, loc, sourceValue, as,
                                            elemTok);
        }
      }

      if (isSetFFTsPointerLikeType(rawPtr.getType())) {
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                         ArrayAttr{}, rcU64, ValueRange{rawPtr})
            .getResult(0);
      }

      if (rawPtr.getType() == u64Ty)
        return rawPtr;
      return rewriter.create<emitc::CastOp>(loc, u64Ty, rawPtr).getResult();
    };

    if (op.getSource().getDefiningOp<pto::DeclareTileMemRefOp>()) {
      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();
      rewriter.replaceOp(op, buildTileValue(*tileSpec));
      return success();
    }

    Value tileCandidate = peelAllCasts(adaptor.getSource());
    if (viewSemantics && viewSemantics.getValue() == "bitcast" &&
        isTileLike(tileCandidate)) {
      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();
      Value dstTile = buildTileValue(*tileSpec);
      FailureOr<Value> addr = buildIntegralAddress(tileCandidate);
      if (failed(addr))
        return failure();

      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{dstTile, *addr});
      rewriter.replaceOp(op, dstTile);
      return success();
    }

    if (viewSemantics && viewSemantics.getValue() == "treshape" &&
        isTileLike(tileCandidate)) {
      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();
      Value dstTile = buildTileValue(*tileSpec, /*forceDeclaration=*/true);

      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TRESHAPE",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{dstTile, tileCandidate});
      rewriter.replaceOp(op, dstTile);
      return success();
    }

    // Subview origins are kept distinct from generic tile rebinding:
    // even when source/destination C++ tile types match, subview may carry
    // shifted base address semantics and should materialize a fresh handle.
    if (isSubView) {
      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();
      Value dstTile = buildTileValue(*tileSpec);
      FailureOr<Value> addr = buildIntegralAddress(tileCandidate);
      if (failed(addr))
        return failure();

      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{dstTile, *addr});
      rewriter.replaceOp(op, dstTile);
      return success();
    }

    // Generic tile-to-tile rebind path: preserve the same backing storage and
    // rebuild a sibling tile with updated metadata/valid dims.
    if (isTileLike(tileCandidate)) {
      FailureOr<TileBuildSpec> tileSpec = buildTileSpec();
      if (failed(tileSpec))
        return failure();

      if (!tileSpec->useConstructor) {
        if (auto srcTy = dyn_cast<emitc::OpaqueType>(tileCandidate.getType())) {
          if (srcTy.getValue() == tileSpec->tileTypeStr) {
            rewriter.replaceOp(op, tileCandidate);
            return success();
          }
        }
      }

      Value dstTile = buildTileValue(*tileSpec);
      FailureOr<Value> addr = buildIntegralAddress(tileCandidate);
      if (failed(addr))
        return failure();

      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{dstTile, *addr});
      rewriter.replaceOp(op, dstTile);
      return success();
    }

    SmallVector<Value> physAddrs;
    Value source = op.getSource();

    while (auto castOp = source.getDefiningOp<UnrealizedConversionCastOp>())
      source = castOp.getOperand(0);

    if (auto upstreamCast = source.getDefiningOp<pto::PointerCastOp>()) {
      auto upstreamOperands = upstreamCast.getAddrs();
      physAddrs.append(upstreamOperands.begin(), upstreamOperands.end());
    } else {
      physAddrs.push_back(adaptor.getSource());
    }

    Value vRow = op.getValidRow();
    Value vCol = op.getValidCol();

    auto newCast = rewriter.create<pto::PointerCastOp>(
        loc, op.getType(), physAddrs, vRow ? vRow : Value(),
        vCol ? vCol : Value(), configAttr);
    if (viewSemantics)
      newCast->setAttr("pto.view_semantics", viewSemantics);
    if (op->hasAttr(kForceDynamicValidShapeAttrName))
      newCast->setAttr(kForceDynamicValidShapeAttrName,
                       op->getAttr(kForceDynamicValidShapeAttrName));
    rewriter.replaceOp(op, newCast.getResult());

    return success();
  }
};

struct PTOAllocTileToEmitC
    : public OpConversionPattern<pto::AllocTileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AllocTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto tileTy = cast<pto::TileBufType>(op.getResult().getType());
    auto tileTypeString = getEmitCTileTypeString(tileTy);
    if (!tileTypeString)
      return rewriter.notifyMatchFailure(
          op, "only rank-2 alloc_tile handles can be converted to EmitC");

    Type convertedTy = getTypeConverter()->convertType(tileTy);
    if (!convertedTy)
      convertedTy = emitc::OpaqueType::get(ctx, *tileTypeString);

    auto validShape = tileTy.getValidShape();
    bool hasDynamicValidDim =
        llvm::any_of(validShape, [](int64_t dim) { return dim < 0; });
    bool useConstructor = hasDynamicValidDim;

    SmallVector<Value> constructorArgs;
    if (useConstructor) {
      Type elemTy = tileTy.getElementType();
      pto::BLayout blayout = getTileBufBLayoutValue(tileTy.getConfigAttr());
      auto maybeScaleDynamicValid = [&](Value emitted, int dimIdx) -> Value {
        if (!emitted || !pto::isPTOFloat4PackedType(elemTy))
          return emitted;
        int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
        if (dimIdx != packedDim)
          return emitted;
        auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
        Value two = makeEmitCIntConstant(rewriter, loc, i32Ty, 2);
        return rewriter.create<emitc::MulOp>(loc, i32Ty, emitted, two)
            .getResult();
      };

      if (validShape.size() > 0 && validShape[0] < 0) {
        Value validRow = adaptor.getValidRow();
        if (!validRow)
          return rewriter.notifyMatchFailure(
              op, "dynamic alloc_tile valid row must have an operand");
        if (validRow)
          validRow = peelUnrealized(validRow);
        constructorArgs.push_back(maybeScaleDynamicValid(validRow, 0));
      }
      if (validShape.size() > 1 && validShape[1] < 0) {
        Value validCol = adaptor.getValidCol();
        if (!validCol)
          return rewriter.notifyMatchFailure(
              op, "dynamic alloc_tile valid col must have an operand");
        if (validCol)
          validCol = peelUnrealized(validCol);
        constructorArgs.push_back(maybeScaleDynamicValid(validCol, 1));
      }
    }

    Value tile;
    if (useConstructor) {
      tile = rewriter
                 .create<emitc::CallOpaqueOp>(
                     loc, convertedTy, *tileTypeString, ArrayAttr{},
                     ArrayAttr{}, ValueRange(constructorArgs))
                 .getResult(0);
    } else {
      tile =
          rewriter
              .create<emitc::VariableOp>(
                  loc, convertedTy, emitc::OpaqueAttr::get(ctx, ""))
              .getResult();
    }

    Value addr = adaptor.getAddr();
    if (addr) {
      addr = peelUnrealized(addr);
      auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
      if (isa<emitc::PointerType>(addr.getType()) ||
          (isa<emitc::OpaqueType>(addr.getType()) &&
           cast<emitc::OpaqueType>(addr.getType()).getValue().ends_with("*"))) {
        auto rcU64 =
            rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
        addr = rewriter
                   .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                                ArrayAttr{}, rcU64,
                                                ValueRange{addr})
                   .getResult(0);
      } else if (addr.getType() != u64Ty) {
        addr = rewriter.create<emitc::CastOp>(loc, u64Ty, addr).getResult();
      }

      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{tile, addr});
    }

    rewriter.replaceOp(op, tile);
    return success();
  }
};

static FailureOr<Value>
createEmitCTileVariable(ConversionPatternRewriter &rewriter, Location loc,
                        const TypeConverter *typeConverter,
                        pto::TileBufType tileTy) {
  auto tileTypeString = getEmitCTileTypeString(tileTy);
  if (!tileTypeString)
    return failure();

  Type convertedTy = typeConverter->convertType(tileTy);
  if (!convertedTy)
    convertedTy = emitc::OpaqueType::get(rewriter.getContext(), *tileTypeString);

  return rewriter
      .create<emitc::VariableOp>(
          loc, convertedTy, emitc::OpaqueAttr::get(rewriter.getContext(), ""))
      .getResult();
}

struct PTOTReshapeToEmitC : public OpConversionPattern<pto::TReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tileTy = dyn_cast<pto::TileBufType>(op.getResult().getType());
    if (!tileTy)
      return failure();

    FailureOr<Value> dst =
        createEmitCTileVariable(rewriter, op.getLoc(), getTypeConverter(), tileTy);
    if (failed(dst))
      return failure();

    Value src = peelUnrealized(adaptor.getSrc());
    if (auto castOp = src.getDefiningOp<emitc::CastOp>())
      src = castOp.getOperand();

    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, "TRESHAPE",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{*dst, src});
    rewriter.replaceOp(op, *dst);
    return success();
  }
};

struct PTOBitcastToEmitC : public OpConversionPattern<pto::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::BitcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto dstTy = dyn_cast<pto::TileBufType>(op.getResult().getType());
    auto srcTy = dyn_cast<pto::TileBufType>(op.getSrc().getType());
    if (!dstTy || !srcTy)
      return failure();

    FailureOr<Value> dst =
        createEmitCTileVariable(rewriter, op.getLoc(), getTypeConverter(), dstTy);
    if (failed(dst))
      return failure();

    Value src = peelUnrealized(adaptor.getSrc());
    if (auto castOp = src.getDefiningOp<emitc::CastOp>())
      src = castOp.getOperand();

    pto::AddressSpace as = pto::AddressSpace::GM;
    if (auto asAttr =
            dyn_cast_or_null<pto::AddressSpaceAttr>(srcTy.getMemorySpace()))
      as = asAttr.getAddressSpace();
    std::string elemTok = getEmitCScalarTypeToken(srcTy.getElementType());

    Value rawPtr = materializeTileDataValue(rewriter, op.getLoc(), src, as, elemTok);
    auto u64Ty = emitc::OpaqueType::get(rewriter.getContext(), "uint64_t");
    Value addr = rawPtr;
    if (isSetFFTsPointerLikeType(rawPtr.getType())) {
      auto rcU64 =
          rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                        "uint64_t")});
      addr = rewriter
                 .create<emitc::CallOpaqueOp>(op.getLoc(), u64Ty,
                                              "reinterpret_cast", ArrayAttr{},
                                              rcU64, ValueRange{rawPtr})
                 .getResult(0);
    } else if (addr.getType() != u64Ty) {
      addr = rewriter.create<emitc::CastOp>(op.getLoc(), u64Ty, addr).getResult();
    }

    rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{}, "TASSIGN",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{*dst, addr});
    rewriter.replaceOp(op, *dst);
    return success();
  }
};

struct PTOMaterializeTileToEmitC
    : public OpConversionPattern<pto::MaterializeTileOp> {
  using OpConversionPattern::OpConversionPattern;

  static bool isTileLike(Value v) {
    auto ot = dyn_cast<emitc::OpaqueType>(v.getType());
    if (!ot)
      return false;
    StringRef s = ot.getValue();
    return s.contains("Tile<") || s.contains("ConvTile<");
  }

  LogicalResult matchAndRewrite(pto::MaterializeTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto tileTy = cast<pto::TileBufType>(op.getResult().getType());
    auto tileTypeString = getEmitCTileTypeString(tileTy);
    if (!tileTypeString)
      return rewriter.notifyMatchFailure(
          op, "only rank-2 tile_buf handles can be materialized to EmitC");

    Type convertedTy = getTypeConverter()->convertType(tileTy);
    if (!convertedTy)
      convertedTy = emitc::OpaqueType::get(ctx, *tileTypeString);

    Value source = peelUnrealized(adaptor.getSource());
    if (auto castOp = source.getDefiningOp<emitc::CastOp>())
      source = castOp.getOperand();

    auto viewSemantics = op->getAttrOfType<StringAttr>("pto.view_semantics");
    bool forceDynamicValid = op->hasAttr(kForceDynamicValidShapeAttrName);
    bool isReshape = viewSemantics && viewSemantics.getValue() == "treshape";
    bool isSubview = viewSemantics && viewSemantics.getValue() == "subview";
    bool sourceIsDeclaredTile =
        op.getSource().getDefiningOp<pto::DeclareTileMemRefOp>();

    auto createTileValue = [&]() -> Value {
      SmallVector<Value, 2> constructorArgs;
      bool useConstructor = false;
      pto::BLayout blayout = getTileBufBLayoutValue(tileTy.getConfigAttr());
      Type elemTy = tileTy.getElementType();
      auto shape = tileTy.getShape();
      auto validShape = tileTy.getValidShape();

      auto makeCtorDimValue = [&](Value emitted, int64_t fallback) -> Value {
        if (emitted)
          return emitted;
        return makeEmitCIntConstant(
            rewriter, loc, emitc::OpaqueType::get(ctx, "int32_t"), fallback);
      };
      auto maybeScaleDynamicValid = [&](Value emitted, int dimIdx) -> Value {
        if (!emitted || !pto::isPTOFloat4PackedType(elemTy))
          return emitted;
        int packedDim = blayout == pto::BLayout::ColMajor ? 0 : 1;
        if (dimIdx != packedDim)
          return emitted;
        auto i32Ty = emitc::OpaqueType::get(ctx, "int32_t");
        Value two = makeEmitCIntConstant(rewriter, loc, i32Ty, 2);
        return rewriter.create<emitc::MulOp>(loc, i32Ty, emitted, two).getResult();
      };
      auto fallbackDim = [&](int dimIdx) {
        return renderTileTemplateDim(shape[dimIdx], elemTy, blayout, dimIdx);
      };

      if (forceDynamicValid) {
        useConstructor = true;
        constructorArgs.push_back(makeCtorDimValue(
            maybeScaleDynamicValid(adaptor.getValidRow(), 0), fallbackDim(0)));
        constructorArgs.push_back(makeCtorDimValue(
            maybeScaleDynamicValid(adaptor.getValidCol(), 1), fallbackDim(1)));
      } else {
        if (validShape[0] == ShapedType::kDynamic) {
          useConstructor = true;
          constructorArgs.push_back(makeCtorDimValue(
              maybeScaleDynamicValid(adaptor.getValidRow(), 0), fallbackDim(0)));
        }
        if (validShape[1] == ShapedType::kDynamic) {
          useConstructor = true;
          constructorArgs.push_back(makeCtorDimValue(
              maybeScaleDynamicValid(adaptor.getValidCol(), 1), fallbackDim(1)));
        }
      }

      if (useConstructor) {
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, convertedTy, *tileTypeString,
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange(constructorArgs))
            .getResult(0);
      }

      return rewriter
          .create<emitc::VariableOp>(loc, convertedTy,
                                     emitc::OpaqueAttr::get(ctx, ""))
          .getResult();
    };

    if (!isSubview && !forceDynamicValid && isTileLike(source)) {
      if (auto srcTy = dyn_cast<emitc::OpaqueType>(source.getType())) {
        if (srcTy.getValue() == *tileTypeString) {
          rewriter.replaceOp(op, source);
          return success();
        }
      }
    }

    Value tile = createTileValue();
    if (sourceIsDeclaredTile) {
      rewriter.replaceOp(op, tile);
      return success();
    }

    if (isReshape && isTileLike(source)) {
      rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TRESHAPE",
                                           ArrayAttr{}, ArrayAttr{},
                                           ValueRange{tile, source});
      rewriter.replaceOp(op, tile);
      return success();
    }

    pto::AddressSpace as = pto::AddressSpace::GM;
    if (auto asAttr =
            dyn_cast_or_null<pto::AddressSpaceAttr>(tileTy.getMemorySpace()))
      as = asAttr.getAddressSpace();
    std::string elemTok = getEmitCScalarTypeToken(tileTy.getElementType());

    Value rawPtr = source;
    if (isTileLike(rawPtr))
      rawPtr = materializeTileDataValue(rewriter, loc, rawPtr, as, elemTok);

    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
    Value addr = rawPtr;
    if (isSetFFTsPointerLikeType(rawPtr.getType())) {
      auto rcU64 =
          rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
      addr = rewriter
                 .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                              ArrayAttr{}, rcU64,
                                              ValueRange{rawPtr})
                 .getResult(0);
    } else if (rawPtr.getType() != u64Ty) {
      addr = rewriter.create<emitc::CastOp>(loc, u64Ty, rawPtr).getResult();
    }

    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                         ArrayAttr{}, ArrayAttr{},
                                         ValueRange{tile, addr});
    rewriter.replaceOp(op, tile);
    return success();
  }
};

// =============================================================================
// Arith CmpI -> EmitC Cmp
// =============================================================================
class ArithCmpIToEmitC : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // 将 arith.cmpi 转换为 emitc.cmp
    // 映射 Predicate: eq -> equal, slt -> less, etc.
    emitc::CmpPredicate emitcPred = emitc::CmpPredicate::eq;
    const bool isUnsignedPred =
        op.getPredicate() == arith::CmpIPredicate::ult ||
        op.getPredicate() == arith::CmpIPredicate::ule ||
        op.getPredicate() == arith::CmpIPredicate::ugt ||
        op.getPredicate() == arith::CmpIPredicate::uge;
    switch (op.getPredicate()) {
      case arith::CmpIPredicate::eq:  emitcPred = emitc::CmpPredicate::eq; break;
      case arith::CmpIPredicate::ne:  emitcPred = emitc::CmpPredicate::ne; break;
      case arith::CmpIPredicate::slt: emitcPred = emitc::CmpPredicate::lt; break;
      case arith::CmpIPredicate::sle: emitcPred = emitc::CmpPredicate::le; break;
      case arith::CmpIPredicate::sgt: emitcPred = emitc::CmpPredicate::gt; break;
      case arith::CmpIPredicate::sge: emitcPred = emitc::CmpPredicate::ge; break;
      // ... 处理无符号比较 (ult, ule 等) ...
      case arith::CmpIPredicate::ult: emitcPred = emitc::CmpPredicate::lt; break;
      case arith::CmpIPredicate::ule: emitcPred = emitc::CmpPredicate::le; break;
      case arith::CmpIPredicate::ugt: emitcPred = emitc::CmpPredicate::gt; break;
      case arith::CmpIPredicate::uge: emitcPred = emitc::CmpPredicate::ge; break;
    }

    Type resTy = getTypeConverter()->convertType(op.getType());
    if (!resTy)
      return failure();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (isUnsignedPred) {
      Type opTy = op.getLhs().getType();
      auto intTy = dyn_cast<IntegerType>(opTy);
      const bool isIndex = isa<IndexType>(opTy);
      if (!intTy && !isIndex)
        return rewriter.notifyMatchFailure(
            op, "expected scalar integer or index operands");

      const unsigned bitWidth =
          intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);
      if (bitWidth != 1) {
        lhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, lhs, bitWidth);
        rhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, rhs, bitWidth);
      }
    }

    rewriter.replaceOpWithNewOp<emitc::CmpOp>(
        op, 
        /*resultType=*/resTy, // i1 -> bool/i1
        emitcPred,
        lhs,
        rhs
    );
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Section Op Lowering
//===----------------------------------------------------------------------===//
static bool isA5NoSplitPipeOp(Operation *op) {
  if (auto talloc = dyn_cast<pto::TAllocOp>(op))
    return talloc.getSplit() == 0;
  if (auto tpush = dyn_cast<pto::TPushOp>(op))
    return tpush.getSplit() == 0;
  if (auto tpop = dyn_cast<pto::TPopOp>(op))
    return tpop.getSplit() == 0;
  if (auto tfree = dyn_cast<pto::TFreeOp>(op))
    return tfree.getSplit() == 0;
  if (auto tpush = dyn_cast<pto::TPushToAivOp>(op))
    return tpush.getSplit() == 0;
  if (auto tpush = dyn_cast<pto::TPushToAicOp>(op))
    return tpush.getSplit() == 0;
  if (auto talloc = dyn_cast<pto::TAllocToAivOp>(op))
    return talloc.getSplit() == 0;
  if (auto talloc = dyn_cast<pto::TAllocToAicOp>(op))
    return talloc.getSplit() == 0;
  if (auto tpop = dyn_cast<pto::TPopFromAicOp>(op))
    return tpop.getSplit() == 0;
  if (auto tpop = dyn_cast<pto::TPopFromAivOp>(op))
    return tpop.getSplit() == 0;
  if (auto tfree = dyn_cast<pto::TFreeFromAicOp>(op))
    return tfree.getSplit() == 0;
  if (auto tfree = dyn_cast<pto::TFreeFromAivOp>(op))
    return tfree.getSplit() == 0;
  return false;
}

static bool hasExplicitSubblockControl(Operation *op) {
  bool hasControl = false;
  op->walk([&](Operation *nested) {
    if (isa<pto::GetSubBlockIdxOp, pto::GetSubBlockNumOp>(nested)) {
      hasControl = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasControl;
}

static bool needsA5NoSplitVectorGuard(Operation *op) {
  auto arch = getTargetArch(op);
  if (arch != PTOArch::A5)
    return false;
  bool isVectorScope = isa<pto::SectionVectorOp>(op);
  if (auto func = dyn_cast<func::FuncOp>(op)) {
    if (auto kernelKindAttr =
            func->getAttrOfType<FunctionKernelKindAttr>(
                FunctionKernelKindAttr::name)) {
      isVectorScope =
          kernelKindAttr.getKernelKind() == FunctionKernelKind::Vector;
    }
  }
  if (!isVectorScope)
    return false;
  if (hasExplicitSubblockControl(op))
    return false;

  bool hasNoSplitPipe = false;
  op->walk([&](Operation *nested) {
    if (!isA5NoSplitPipeOp(nested))
      return WalkResult::advance();
    hasNoSplitPipe = true;
    return WalkResult::interrupt();
  });
  return hasNoSplitPipe;
}

template <typename SectionOpTy>
struct SectionToEmitC : public OpConversionPattern<SectionOpTy> {
  using OpConversionPattern<SectionOpTy>::OpConversionPattern;

  std::string getMacroName() const {
    if (std::is_same<SectionOpTy, pto::SectionCubeOp>::value)
      return "__DAV_CUBE__";
    if (std::is_same<SectionOpTy, pto::SectionVectorOp>::value)
      return "__DAV_VEC__";
    return "UNKNOWN_MACRO";
  }

  LogicalResult
  matchAndRewrite(SectionOpTy op, typename SectionOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    bool needsNoSplitGuard = needsA5NoSplitVectorGuard(op.getOperation());

    std::string startMacro = "\n#if defined(" + getMacroName() + ")";
    rewriter.create<emitc::VerbatimOp>(loc, startMacro);

    if constexpr (std::is_same_v<SectionOpTy, pto::SectionVectorOp>) {
      // Vector mask is a global HW state and may be modified by previous kernels
      // (or earlier sections). Reset it to a well-defined state for deterministic
      // execution of VEC ops.
      rewriter.create<emitc::VerbatimOp>(loc, "set_mask_norm();");
      rewriter.create<emitc::VerbatimOp>(loc, "set_vector_mask(-1, -1);");
    }

    if (needsNoSplitGuard) {
      rewriter.create<emitc::VerbatimOp>(
          loc, "if (get_subblockid() == 0) {");
    }

    Block &innerBlock = op.getBody().front();
    if (!innerBlock.empty()) {
      rewriter.inlineBlockBefore(&innerBlock, op.getOperation(), ValueRange{});
    }

    if (needsNoSplitGuard)
      rewriter.create<emitc::VerbatimOp>(loc, "}");

    std::string endMacro = "#endif // " + getMacroName() + "\n";
    rewriter.create<emitc::VerbatimOp>(loc, endMacro);

    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// SCF Control-Flow Pre-Lowering
//
// EmitC translation supports `emitc.for`/`emitc.if` plus CFG-style
// `cf.br`/`cf.cond_br`. Upstream SCFToEmitC patterns only cover `scf.for` and
// `scf.if`, so we pre-lower some SCF ops into those supported forms.
//===----------------------------------------------------------------------===//

namespace {

static bool isTriviallyInlineableExecuteRegion(scf::ExecuteRegionOp op) {
  Region &r = op.getRegion();
  if (!r.hasOneBlock())
    return false;
  Block &b = r.front();
  return isa_and_nonnull<scf::YieldOp>(b.getTerminator());
}

static bool needsWholeFunctionSCFToCF(func::FuncOp func) {
  bool needs = false;
  func.walk([&](Operation *op) {
    if (!isa<scf::WhileOp, scf::IndexSwitchOp, scf::ExecuteRegionOp>(op))
      return WalkResult::advance();
    Operation *parentOp = op->getParentOp();

    // `scf.execute_region` can legally appear in single-block parents. Only
    // require whole-function SCFToCF if we need to lower it into CFG blocks
    // (multi-block region / non-trivial terminators).
    if (auto exec = dyn_cast<scf::ExecuteRegionOp>(op)) {
      if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>() &&
          !isTriviallyInlineableExecuteRegion(exec)) {
        needs = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      needs = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return needs;
}

// scf.execute_region is semantically just an inlined region producing results
// via scf.yield. Inline it to the parent block to avoid extra lowering needs.
struct SCFExecuteRegionInline
    : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getRegion().empty())
      return rewriter.notifyMatchFailure(op, "expected non-empty region");

    Block &innerBlock = op.getRegion().front();
    auto yield = dyn_cast<scf::YieldOp>(innerBlock.getTerminator());
    if (!yield)
      return rewriter.notifyMatchFailure(op, "expected scf.yield terminator");

    // Move the body operations before the execute_region op.
    rewriter.inlineBlockBefore(&innerBlock, op.getOperation(), ValueRange{});

    // Replace execute_region results with yielded values, then erase the yield.
    rewriter.replaceOp(op, yield.getOperands());
    rewriter.eraseOp(yield);
    return success();
  }
};

// Lower scf.execute_region into CFG blocks with cf.br/cf.cond_br by inlining the
// region blocks into the parent region and rewriting scf.yield to branch into a
// continuation block carrying results.
//
// Note: This requires the parent region to allow multiple blocks (e.g. the
// function body CFG region). For execute_region nested in single-block regions
// (scf.for/scf.if), run SCFToCF first to eliminate the single-block constraint.
struct SCFExecuteRegionToCF : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (isTriviallyInlineableExecuteRegion(op))
      return rewriter.notifyMatchFailure(op, "trivially inlineable");

    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower scf.execute_region inside a single-block parent region");
    }

    if (op.getRegion().empty())
      return rewriter.notifyMatchFailure(op, "expected non-empty region");

    Location loc = op.getLoc();
    Block *curBlock = op->getBlock();
    Region *parentRegion = curBlock->getParent();

    // Split the parent block so we can branch to a continuation block with phi
    // arguments for the execute_region results.
    auto execIt = Block::iterator(op.getOperation());
    Block *continueBlock = rewriter.splitBlock(curBlock, std::next(execIt));

    SmallVector<BlockArgument> contArgs;
    contArgs.reserve(op.getNumResults());
    for (Type t : op.getResultTypes())
      contArgs.push_back(continueBlock->addArgument(t, loc));

    for (auto it : llvm::enumerate(op.getResults()))
      it.value().replaceAllUsesWith(contArgs[it.index()]);

    // Capture blocks before moving the region.
    SmallVector<Block *> movedBlocks;
    movedBlocks.reserve(op.getRegion().getBlocks().size());
    for (Block &b : op.getRegion())
      movedBlocks.push_back(&b);
    Block *entryBlock = &op.getRegion().front();

    // Inline the execute_region blocks into the parent region right before the
    // continuation block.
    rewriter.inlineRegionBefore(op.getRegion(), *parentRegion,
                                continueBlock->getIterator());

    // Replace all scf.yield terminators with a branch to the continuation.
    for (Block *b : movedBlocks) {
      auto yield = dyn_cast<scf::YieldOp>(b->getTerminator());
      if (!yield)
        continue;
      rewriter.setInsertionPoint(yield);
      rewriter.create<cf::BranchOp>(loc, continueBlock, yield.getOperands());
      rewriter.eraseOp(yield);
    }

    // Replace execute_region itself with a branch to the inlined entry block.
    rewriter.setInsertionPoint(op);
    rewriter.create<cf::BranchOp>(loc, entryBlock, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower scf.index_switch into CFG blocks with cf.cond_br/cf.br so that we can
// avoid `scf.if` result materialization quirks (and avoid relying on cf.switch,
// which is not supported by EmitC C++ translation).
struct SCFIndexSwitchToCF : public OpRewritePattern<scf::IndexSwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  static LogicalResult cloneYieldingBlockAndBranchTo(
      PatternRewriter &rewriter, Location loc, Block &srcBlock, Block *destBlock,
      Block *continueBlock) {
    rewriter.setInsertionPointToEnd(destBlock);

    IRMapping mapping;
    for (Operation &inner : srcBlock.without_terminator())
      rewriter.clone(inner, mapping);

    auto yield = dyn_cast<scf::YieldOp>(srcBlock.getTerminator());
    if (!yield)
      return failure();

    SmallVector<Value> yieldOperands;
    yieldOperands.reserve(yield.getNumOperands());
    for (Value v : yield.getOperands())
      yieldOperands.push_back(mapping.lookupOrDefault(v));

    rewriter.create<cf::BranchOp>(loc, continueBlock, yieldOperands);
    return success();
  }

  static Block *splitBlockForContinuation(PatternRewriter &rewriter,
                                          scf::IndexSwitchOp op) {
    auto switchIt = Block::iterator(op.getOperation());
    return rewriter.splitBlock(op->getBlock(), std::next(switchIt));
  }

  static void addContinuationArguments(PatternRewriter &rewriter,
                                       scf::IndexSwitchOp op, Location loc,
                                       Block *continueBlock) {
    SmallVector<BlockArgument> contArgs;
    contArgs.reserve(op.getNumResults());
    for (Type type : op.getResultTypes())
      contArgs.push_back(continueBlock->addArgument(type, loc));
    for (auto result : llvm::enumerate(op.getResults()))
      result.value().replaceAllUsesWith(contArgs[result.index()]);
  }

  static void createIndexSwitchBlocks(PatternRewriter &rewriter,
                                      Region *parentRegion,
                                      Region::iterator insertPt,
                                      unsigned numCases,
                                      SmallVectorImpl<Block *> &checkBlocks,
                                      Block *&defaultBlock,
                                      SmallVectorImpl<Block *> &caseBlocks) {
    checkBlocks.reserve(numCases);
    caseBlocks.reserve(numCases);
    for (unsigned i = 0; i < numCases; ++i)
      checkBlocks.push_back(rewriter.createBlock(parentRegion, insertPt));
    defaultBlock = rewriter.createBlock(parentRegion, insertPt);
    for (unsigned i = 0; i < numCases; ++i)
      caseBlocks.push_back(rewriter.createBlock(parentRegion, insertPt));
  }

  static void populateIndexSwitchCheckBlocks(
      PatternRewriter &rewriter, Location loc, Value selector,
      ArrayRef<int64_t> cases, ArrayRef<Block *> checkBlocks,
      ArrayRef<Block *> caseBlocks, Block *defaultBlock) {
    for (unsigned i = 0; i < checkBlocks.size(); ++i) {
      rewriter.setInsertionPointToEnd(checkBlocks[i]);
      Value caseVal = rewriter.create<arith::ConstantIndexOp>(loc, cases[i]);
      Value cond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, selector, caseVal);
      Block *falseDest =
          (i + 1 < checkBlocks.size()) ? checkBlocks[i + 1] : defaultBlock;
      rewriter.create<cf::CondBranchOp>(loc, cond, caseBlocks[i], ValueRange{},
                                        falseDest, ValueRange{});
    }
  }

  LogicalResult matchAndRewrite(scf::IndexSwitchOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower scf.index_switch inside a single-block parent region");
    }

    Block *curBlock = op->getBlock();
    Region *parentRegion = curBlock->getParent();
    Block *continueBlock = splitBlockForContinuation(rewriter, op);
    addContinuationArguments(rewriter, op, loc, continueBlock);

    unsigned numCases = op.getCases().size();
    auto insertPt = continueBlock->getIterator();

    SmallVector<Block *> checkBlocks;
    SmallVector<Block *> caseBlocks;
    Block *defaultBlock = nullptr;
    createIndexSwitchBlocks(rewriter, parentRegion, insertPt, numCases,
                            checkBlocks, defaultBlock, caseBlocks);

    Value selector = op.getArg();
    auto cases = op.getCases();
    populateIndexSwitchCheckBlocks(rewriter, loc, selector, cases, checkBlocks,
                                   caseBlocks, defaultBlock);

    // Fill case blocks and default block with cloned bodies + branch to cont.
    for (unsigned i = 0; i < numCases; ++i) {
      if (failed(cloneYieldingBlockAndBranchTo(
              rewriter, loc, op.getCaseBlock(i), caseBlocks[i], continueBlock)))
        return rewriter.notifyMatchFailure(op, "expected scf.yield terminator");
    }
    if (failed(cloneYieldingBlockAndBranchTo(rewriter, loc, op.getDefaultBlock(),
                                             defaultBlock, continueBlock)))
      return rewriter.notifyMatchFailure(op, "expected scf.yield terminator");

    // Replace the original switch op with a branch into the check chain.
    Block *entryDest = numCases ? checkBlocks[0] : defaultBlock;
    rewriter.setInsertionPointAfter(op);
    rewriter.create<cf::BranchOp>(loc, entryDest, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower scf.while into CFG blocks with cf.br/cf.cond_br.
//
// Note: This requires the parent region to allow multiple blocks. In
// particular, scf.if/scf.for regions are single-block and cannot contain this
// lowering.
struct SCFWhileToCF : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  static LogicalResult validateWhileResultUses(scf::WhileOp op) {
    Block *parentBlock = op->getBlock();
    for (Value result : op.getResults()) {
      for (OpOperand &use : result.getUses()) {
        if (use.getOwner()->getBlock() != parentBlock)
          return failure();
      }
    }
    return success();
  }

  static Block *splitAfterWhileBlock(PatternRewriter &rewriter,
                                     scf::WhileOp op) {
    auto whileIt = Block::iterator(op.getOperation());
    return rewriter.splitBlock(op->getBlock(), std::next(whileIt));
  }

  static void addWhileExitArguments(PatternRewriter &rewriter, scf::WhileOp op,
                                    Location loc, Block *afterWhileBlock) {
    SmallVector<Value> exitArgs;
    exitArgs.reserve(op.getNumResults());
    for (Type type : op.getResultTypes())
      exitArgs.push_back(afterWhileBlock->addArgument(type, loc));
    for (auto result : llvm::enumerate(op.getResults()))
      result.value().replaceAllUsesWith(exitArgs[result.index()]);
  }

  static Block *createWhileHeaderBlock(PatternRewriter &rewriter,
                                       scf::WhileOp op, Location loc,
                                       Block *afterWhileBlock) {
    SmallVector<Type> headerArgTypes;
    for (Value init : op.getInits())
      headerArgTypes.push_back(init.getType());
    SmallVector<Location> headerArgLocs(headerArgTypes.size(), loc);
    return rewriter.createBlock(afterWhileBlock->getParent(),
                                afterWhileBlock->getIterator(), headerArgTypes,
                                headerArgLocs);
  }

  static Block *createWhileBodyBlock(PatternRewriter &rewriter, scf::WhileOp op,
                                     Location loc, Block *afterWhileBlock) {
    Block &afterRegionBlock = op.getAfter().front();
    SmallVector<Type> bodyArgTypes(afterRegionBlock.getArgumentTypes().begin(),
                                   afterRegionBlock.getArgumentTypes().end());
    SmallVector<Location> bodyArgLocs(bodyArgTypes.size(), loc);
    return rewriter.createBlock(afterWhileBlock->getParent(),
                                afterWhileBlock->getIterator(), bodyArgTypes,
                                bodyArgLocs);
  }

  static void rewriteWhileTerminators(PatternRewriter &rewriter, Location loc,
                                      Block *headerBlock, Block *bodyBlock,
                                      Block *afterWhileBlock) {
    auto condOp = cast<scf::ConditionOp>(headerBlock->getTerminator());
    rewriter.setInsertionPoint(condOp);
    rewriter.create<cf::CondBranchOp>(loc, condOp.getCondition(),
                                      /*trueDest=*/bodyBlock,
                                      /*trueOperands=*/condOp.getArgs(),
                                      /*falseDest=*/afterWhileBlock,
                                      /*falseOperands=*/condOp.getArgs());
    rewriter.eraseOp(condOp);

    auto yieldOp = cast<scf::YieldOp>(bodyBlock->getTerminator());
    rewriter.setInsertionPoint(yieldOp);
    rewriter.create<cf::BranchOp>(loc, headerBlock, yieldOp.getOperands());
    rewriter.eraseOp(yieldOp);
  }

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower scf.while inside a single-block parent region");
    }

    if (failed(validateWhileResultUses(op)))
      return rewriter.notifyMatchFailure(
          op, "unsupported: while results used outside the parent block");

    auto loc = op.getLoc();
    Block *afterWhileBlock = splitAfterWhileBlock(rewriter, op);
    addWhileExitArguments(rewriter, op, loc, afterWhileBlock);
    Block *headerBlock = createWhileHeaderBlock(rewriter, op, loc,
                                                afterWhileBlock);
    Block *bodyBlock = createWhileBodyBlock(rewriter, op, loc, afterWhileBlock);

    // Move the before/after region bodies into the new CFG blocks.
    Block &afterRegionBlock = op.getAfter().front();
    rewriter.mergeBlocks(&op.getBefore().front(), headerBlock,
                         headerBlock->getArguments());
    rewriter.mergeBlocks(&afterRegionBlock, bodyBlock, bodyBlock->getArguments());
    rewriteWhileTerminators(rewriter, loc, headerBlock, bodyBlock,
                            afterWhileBlock);

    // Replace scf.while itself with a branch to the header.
    rewriter.setInsertionPoint(op);
    rewriter.create<cf::BranchOp>(loc, headerBlock, op.getInits());
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower cf.switch into chained comparisons and cf.cond_br/cf.br.
//
// EmitC C++ translation currently supports cf.br/cf.cond_br, but not cf.switch.
struct CFSwitchToCondBr : public OpRewritePattern<cf::SwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  static SmallVector<SmallVector<Value>>
  collectSwitchCaseOperands(cf::SwitchOp op) {
    SmallVector<SmallVector<Value>> caseOperands;
    caseOperands.reserve(op.getCaseDestinations().size());
    for (auto range : op.getCaseOperands())
      caseOperands.emplace_back(range.begin(), range.end());
    return caseOperands;
  }

  static SmallVector<APInt> getSwitchCaseValues(cf::SwitchOp op) {
    SmallVector<APInt> caseValues;
    if (auto caseValuesAttr = op.getCaseValues()) {
      for (APInt value : caseValuesAttr->getValues<APInt>())
        caseValues.push_back(value);
    }
    return caseValues;
  }

  static SmallVector<Block *> createSwitchCheckBlocks(PatternRewriter &rewriter,
                                                      Region *parentRegion,
                                                      Block *curBlock,
                                                      size_t numCases) {
    auto insertPt = std::next(curBlock->getIterator());
    SmallVector<Block *> checkBlocks;
    checkBlocks.reserve(numCases);
    for (size_t i = 0; i < numCases; ++i)
      checkBlocks.push_back(rewriter.createBlock(parentRegion, insertPt));
    return checkBlocks;
  }

  static LogicalResult populateSwitchCheckBlocks(
      PatternRewriter &rewriter, Location loc, Value flag, IntegerType flagTy,
      ArrayRef<APInt> caseValues, ArrayRef<Block *> caseDests,
      ArrayRef<SmallVector<Value>> caseOperands, Block *defaultDest,
      ValueRange defaultOperands, ArrayRef<Block *> checkBlocks,
      cf::SwitchOp op) {
    for (size_t i = 0; i < caseDests.size(); ++i) {
      rewriter.setInsertionPointToEnd(checkBlocks[i]);
      APInt caseVal = caseValues[i];
      if (caseVal.getBitWidth() != flagTy.getWidth()) {
        return rewriter.notifyMatchFailure(
            op, "case value bitwidth doesn't match flag type");
      }

      Value caseConst = rewriter.create<arith::ConstantOp>(
          loc, flagTy, rewriter.getIntegerAttr(flagTy, caseVal));
      Value cond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, flag, caseConst);
      Block *falseDest =
          (i + 1 < checkBlocks.size()) ? checkBlocks[i + 1] : defaultDest;
      ValueRange falseOperands =
          (i + 1 < checkBlocks.size()) ? ValueRange{} : defaultOperands;
      rewriter.create<cf::CondBranchOp>(loc, cond, caseDests[i],
                                        caseOperands[i], falseDest,
                                        falseOperands);
    }
    return success();
  }

  LogicalResult matchAndRewrite(cf::SwitchOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower cf.switch inside a single-block parent region");
    }

    Block *curBlock = op->getBlock();
    Region *parentRegion = curBlock->getParent();

    Value flag = op.getFlag();
    auto flagTy = dyn_cast<IntegerType>(flag.getType());
    if (!flagTy)
      return rewriter.notifyMatchFailure(op, "expected integer switch flag");

    SmallVector<Value> defaultOperands(op.getDefaultOperands().begin(),
                                       op.getDefaultOperands().end());
    Block *defaultDest = op.getDefaultDestination();

    SmallVector<Block *> caseDests(op.getCaseDestinations().begin(),
                                   op.getCaseDestinations().end());
    SmallVector<SmallVector<Value>> caseOperands = collectSwitchCaseOperands(op);

    if (caseDests.empty()) {
      rewriter.replaceOpWithNewOp<cf::BranchOp>(op, defaultDest, defaultOperands);
      return success();
    }

    if (!op.getCaseValues())
      return rewriter.notifyMatchFailure(op, "missing case_values");
    SmallVector<APInt> caseValues = getSwitchCaseValues(op);

    if (caseValues.size() != caseDests.size())
      return rewriter.notifyMatchFailure(op, "case_values/destinations mismatch");
    if (caseOperands.size() != caseDests.size())
      return rewriter.notifyMatchFailure(op, "case_operands/destinations mismatch");

    SmallVector<Block *> checkBlocks =
        createSwitchCheckBlocks(rewriter, parentRegion, curBlock,
                                caseDests.size());
    if (failed(populateSwitchCheckBlocks(rewriter, loc, flag, flagTy,
                                         caseValues, caseDests, caseOperands,
                                         defaultDest, defaultOperands,
                                         checkBlocks, op))) {
      return failure();
    }

    // Replace the switch terminator with a branch into the first check block.
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, checkBlocks.front(),
                                              ValueRange{});
    return success();
  }
};

} // namespace

static void populatePTOToEmitCPatterns(RewritePatternSet &patterns,
                                       TypeConverter &typeConverter,
                                       MLIRContext *ctx,
                                       DataFlowSolver &solver,
                                       PTOArch targetArch) {
  (void)solver;
  patterns.add<ArithCmpIToEmitC>(typeConverter, ctx);
  patterns.add<PTOAllocTileToEmitC>(typeConverter, ctx);
  patterns.add<PTOMaterializeTileToEmitC>(typeConverter, ctx);
  patterns.add<PTOBindTileToEmitC>(typeConverter, ctx);
  patterns.add<PTOSetFlagToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.set_flag_dyn",
                                      "set_flag");
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.wait_flag_dyn",
                                      "wait_flag");
  // Backward-compatible aliases used in some downstream branches.
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.set_flag_d",
                                      "set_flag");
  patterns.add<PTOSyncFlagDynToEmitC>(typeConverter, ctx, "pto.wait_flag_d",
                                      "wait_flag");
  patterns.add<PTOSubSCToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubCSToEmitC>(typeConverter, ctx);
  patterns.add<PTOWaitFlagToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncAllToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetBufToEmitC>(typeConverter, ctx);
  patterns.add<PTORlsBufToEmitC>(typeConverter, ctx);
  patterns.add<PTOSetFFTsToEmitC>(typeConverter, ctx);
  patterns.add<PTOXORSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubSToEmitC>(typeConverter, ctx);
  patterns.add<PTOXORToEmitC>(typeConverter, ctx);
  patterns.add<PTOReluToEmitC>(typeConverter, ctx);
  patterns.add<PTOScatterToEmitC>(typeConverter, ctx);
  patterns.add<PTOStoreFPSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubSSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSqrtSToEmitC>(typeConverter, ctx);
  patterns.add<PTOTTransToEmitC>(typeConverter, ctx);
  patterns.add<PTOSelSToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandAddToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandDivToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandExpdifToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandMulToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOColExpandSubToEmitC>(typeConverter, ctx);
  patterns.add<PTOColMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOColProdToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandAddToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandExpdifToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandMinToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandSubToEmitC>(typeConverter, ctx);
  patterns.add<PTOShrSToEmitC>(typeConverter, ctx);
  patterns.add<PTOShlSToEmitC>(typeConverter, ctx);
  patterns.add<PTOShlSConstToEmitC>(typeConverter, ctx);
  patterns.add<PTOShrSConstToEmitC>(typeConverter, ctx);
  patterns.add<PTOSORT32SToEmitC>(typeConverter, ctx);
  patterns.add<PTOSelToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandToEmitC>(typeConverter, ctx);
  patterns.add<PTORsqrtToEmitC>(typeConverter, ctx);
  patterns.add<PTORowMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTORowArgMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandMulToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandDivToEmitC>(typeConverter, ctx);
  patterns.add<PTORowProdToEmitC>(typeConverter, ctx);
  patterns.add<PTORowSumToEmitC>(typeConverter, ctx);
  patterns.add<PTORowMinToEmitC>(typeConverter, ctx);
  patterns.add<PTORowArgMinToEmitC>(typeConverter, ctx);
  patterns.add<PTODivSToEmitC>(typeConverter, ctx);
  patterns.add<PTOTDivSToEmitC>(typeConverter, ctx);
  patterns.add<PTOFModToEmitC>(typeConverter, ctx);
  patterns.add<PTORemToEmitC>(typeConverter, ctx);
  patterns.add<PTOConcatToEmitC, PTOConcatidxToEmitC>(typeConverter, ctx);
  patterns.add<PTORecipToEmitC>(typeConverter, ctx);
  patterns.add<PTOMulsToEmitC>(typeConverter, ctx);
  patterns.add<PTOExpToEmitC>(typeConverter, ctx);
  patterns.add<PTOPreluToEmitC>(typeConverter, ctx);
  patterns.add<PTOFModSToEmitC>(typeConverter, ctx);
  patterns.add<PTORemSToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTONotToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartArgMaxToEmitC, PTOPartArgMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMulToEmitC>(typeConverter, ctx);
  patterns.add<PTOExpandsToEmitC>(typeConverter, ctx);
  patterns.add<PTOOrToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartAddToEmitC>(typeConverter, ctx);
  patterns.add<PTOExtractToEmitC, PTOExtractFPToEmitC, PTOInsertToEmitC,
               PTOInsertFPToEmitC>(typeConverter, ctx);
  patterns.add<PTOFillPadToEmitC, PTOFillPadInplaceToEmitC, PTOFillPadExpandToEmitC>(
      typeConverter, ctx);
  patterns.add<PTOGatherToEmitC>(typeConverter, ctx);
  patterns.add<PTOGatherbToEmitC>(typeConverter, ctx);
  patterns.add<PTOMovFPToEmitC>(typeConverter, ctx);
  patterns.add<PTOQuantToEmitC>(typeConverter, ctx);
  patterns.add<PTODequantToEmitC>(typeConverter, ctx);
  patterns.add<PTOOrsToEmitC>(typeConverter, ctx);
  patterns.add<PTOLogToEmitC>(typeConverter, ctx);
  patterns.add<FuncToEmitC>(typeConverter, ctx);
  patterns.add<PTOMovToEmitC>(typeConverter, ctx);
  patterns.add<ArithConstantToEmitC>(typeConverter, ctx);
  patterns.add<ArithAddUIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulSIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulUIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<AffineApplyMulConstToEmitC>(typeConverter, ctx);
  patterns.add<PTONegToEmitC>(typeConverter, ctx);
  patterns.add<PTOTCIToEmitC>(typeConverter, ctx);
  patterns.add<PTOTTriToEmitC>(typeConverter, ctx);
  patterns.add<PTOCmpToEmitC>(typeConverter, ctx);
  patterns.add<PTOCmpSToEmitC>(typeConverter, ctx);
  patterns.add<PTOColSumToEmitC>(typeConverter, ctx);
  patterns.add<PTOLReluToEmitC>(typeConverter, ctx);
  patterns.add<PTOMrgSortToEmitC>(typeConverter, ctx);
  patterns.add<PTORandomToEmitC>(typeConverter, ctx);
  patterns.add<SubviewToEmitCPattern>(typeConverter, ctx);
  patterns.add<PointerCastConversion>(typeConverter, ctx);
  patterns.add<PTOSetValToSETVAL, PTOGetValToGETVAL, PTOSetValidShapeToEmitC,
               PTOGetValidShapeToEmitC, PTOTAssignToEmitC,
               PTOPtrToIntToEmitC, PTOIntToPtrToEmitC, PTOLoadScalarToEmitC,
               PTOStoreScalarToEmitC>(typeConverter, ctx);
  patterns.add<PTOTAxpyToEmitC, PTOHistogramToEmitC, PTOGetScaleAddrToEmitC>(
      typeConverter, ctx);
  patterns.add<PTOTAndToEmitC>(typeConverter, ctx);
  patterns.add<PTOMulToEmitC>(typeConverter, ctx);
  patterns.add<PTOAndSToEmitC>(typeConverter, ctx);
  patterns.add<PTOCvtToEmitC>(typeConverter, ctx);
  patterns.add<PTODivToTDIV>(typeConverter, ctx);
  patterns.add<PTOMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOMaxSToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulIToEmitC>(typeConverter, ctx);
  patterns.add<ArithAddIToEmitC>(typeConverter, ctx);
  patterns.add<ArithSubIToEmitC>(typeConverter, ctx);
  patterns.add<ArithUnsignedBitwiseBinaryToEmitC<arith::AndIOp, emitc::BitwiseAndOp>>(
      typeConverter, ctx);
  patterns.add<ArithUnsignedBitwiseBinaryToEmitC<arith::OrIOp, emitc::BitwiseOrOp>>(
      typeConverter, ctx);
  patterns.add<ArithUnsignedBitwiseBinaryToEmitC<arith::XOrIOp, emitc::BitwiseXorOp>>(
      typeConverter, ctx);
  patterns.add<ArithShiftLeftToEmitC>(typeConverter, ctx);
  patterns.add<ArithShiftRightUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithShiftRightSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithDivUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCeilDivUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCeilDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithFloorDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithRemUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithRemSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithNegFToEmitC>(typeConverter, ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::SubFOp, emitc::SubOp>>(typeConverter,
                                                                     ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::MulFOp, emitc::MulOp>>(typeConverter,
                                                                     ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::DivFOp, emitc::DivOp>>(typeConverter,
                                                                     ctx);
  patterns.add<ArithRemFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaximumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinimumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxNumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinNumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithSelectToEmitC>(typeConverter, ctx);
  patterns.add<ArithCmpFToEmitC>(typeConverter, ctx);
  patterns.add<ArithExtUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithExtSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::ExtFOp>>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::TruncFOp>>(typeConverter, ctx);
  patterns.add<ArithUIToFPToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::SIToFPOp>>(typeConverter, ctx);
  patterns.add<ArithFPToUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::FPToSIOp>>(typeConverter, ctx);
  patterns.add<ArithIndexCastUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithBitcastToEmitC>(typeConverter, ctx);
  patterns.add<PTOAddSToTADDS>(typeConverter, ctx);
  patterns.add<PTOColExpandToEmitC>(typeConverter, ctx);
  patterns.add<PTOColArgMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOColMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOColArgMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOTLoadToTLOAD>(typeConverter, ctx);
  patterns.add<PTOTPrefetchToTPREFETCH>(typeConverter, ctx);
  patterns.add<PTOMakePrefetchAsyncContextToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetPrefetchAsyncSessionToEmitC>(typeConverter, ctx);
  patterns.add<PTOTPrefetchAsyncToEmitC>(typeConverter, ctx);
  patterns.add<PTOTStoreToTSTORE>(typeConverter, ctx);
  patterns.add<PTOMScatterToMSCATTER>(typeConverter, ctx);
  patterns.add<PTOTAddCToTADDC>(typeConverter, ctx);
  patterns.add<PTOMinsToEmitC>(typeConverter, ctx);
  patterns.add<PTOMGatherToMGATHER>(typeConverter, ctx);
  patterns.add<PTOTMatmulToTMATMUL>(typeConverter, ctx);
  patterns.add<PTOTMatmulAccToTMATMULACC>(typeConverter, ctx);
  patterns.add<PTOTGemvToTGEMV>(typeConverter, ctx);
  patterns.add<PTOTGemvAccToTGEMVACC>(typeConverter, ctx);
  patterns.add<ReinterpretCastToEmitC>(typeConverter, ctx);
  patterns.add<PTOTAbsToTABS>(typeConverter, ctx);
  patterns.add<PTOTAddToTADD>(typeConverter, ctx);
  patterns.add<PTOAddSCToTADDSC>(typeConverter, ctx);
  patterns.add<ArithCastOPToEmitC>(typeConverter, ctx);
  patterns.add<ArithTruncIToEmitC>(typeConverter, ctx);
  patterns.add<PTOBuildAsyncSessionToEmitC>(typeConverter, ctx);
  patterns.add<PTOAsyncTransferToEmitC<pto::TPutAsyncOp>>(
      typeConverter, ctx,
      "pto::comm::TPUT_ASYNC<pto::comm::DmaEngine::SDMA>");
  patterns.add<PTOAsyncTransferToEmitC<pto::TGetAsyncOp>>(
      typeConverter, ctx,
      "pto::comm::TGET_ASYNC<pto::comm::DmaEngine::SDMA>");
  patterns.add<PTOP2PCommToEmitC<pto::TPutOp>>(typeConverter, ctx,
                                               "pto::comm::TPUT");
  patterns.add<PTOP2PCommToEmitC<pto::TGetOp>>(typeConverter, ctx,
                                               "pto::comm::TGET");
  patterns.add<PTOSignalCommToEmitC<pto::TNotifyOp>>(typeConverter, ctx,
                                                     "pto::comm::TNOTIFY");
  patterns.add<PTOSignalCommToEmitC<pto::TWaitOp>>(typeConverter, ctx,
                                                   "pto::comm::TWAIT");
  patterns.add<PTOSignalCommToEmitC<pto::TTestOp>>(typeConverter, ctx,
                                                   "pto::comm::TTEST");
  patterns.add<PTOCommCollectiveToEmitC<pto::TBroadcastOp>>(typeConverter, ctx,
                                                            "TBROADCAST");
  patterns.add<PTOCommCollectiveToEmitC<pto::CommTGatherOp>>(typeConverter, ctx,
                                                             "TGATHER");
  patterns.add<PTOCommCollectiveToEmitC<pto::CommTScatterOp>>(typeConverter, ctx,
                                                              "TSCATTER");
  patterns.add<PTOCommCollectiveToEmitC<pto::TReduceOp>>(typeConverter, ctx,
                                                         "TREDUCE");
  patterns.add<PTOAsyncEventToEmitC<pto::WaitAsyncEventOp>>(
      typeConverter, ctx, "PTOAS__ASYNC_EVENT_WAIT");
  patterns.add<PTOAsyncEventToEmitC<pto::TestAsyncEventOp>>(
      typeConverter, ctx, "PTOAS__ASYNC_EVENT_TEST");
  patterns.add<PTOInitializeL2G2LPipeToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOInitializeL2LPipeToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTODeclareTileMemRefToEmitC>(typeConverter, ctx);
  patterns.add<PTODeclareGlobalToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartitionViewToEmitC>(typeConverter, ctx);
  patterns.add<PTODeclareEventIdArrayToEmitC>(typeConverter, ctx);
  patterns.add<PTOEventIdArrayGetToEmitC>(typeConverter, ctx);
  patterns.add<PTOEventIdArraySetToEmitC>(typeConverter, ctx);
  patterns.add<PTODeclareLocalArrayToEmitC>(typeConverter, ctx);
  patterns.add<PTOLocalArrayGetToEmitC>(typeConverter, ctx);
  patterns.add<PTOLocalArraySetToEmitC>(typeConverter, ctx);
  patterns.add<PTOTReshapeToEmitC>(typeConverter, ctx);
  patterns.add<PTOBitcastToEmitC>(typeConverter, ctx);
  patterns.add<PTOTAllocToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOTPushToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOTPopToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOTFreeToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOSyncSetToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<PTOSyncWaitToEmitC>(typeConverter, ctx, targetArch);
  patterns.add<SectionToEmitC<pto::SectionCubeOp>>(typeConverter, ctx);
  patterns.add<SectionToEmitC<pto::SectionVectorOp>>(typeConverter, ctx);
  patterns.add<PTOGetBlockIdxToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetBlockNumToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetSubBlockIdxToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetSubBlockNumToEmitC>(typeConverter, ctx);
  patterns.add<PTOPrintToTPRINT>(typeConverter, ctx);
  patterns.add<PTOPrintOpToEmitC>(typeConverter, ctx);
  patterns.add<PTOTrapOpToEmitC>(typeConverter, ctx);
  patterns.add<
    PTOTMatmulBiasToTMATMUL_BIAS,
    PTOTMatmulMXToTMATMUL_MX,
    PTOTMatmulMXAccToTMATMUL_MX_ACC,
    PTOTMatmulMXBiasToTMATMUL_MX_BIAS,
    PTOTMatmulBiasToTMATMUL_BIAS,
    PTOTMatmulMXToTMATMUL_MX,
    PTOTMatmulMXAccToTMATMUL_MX_ACC,
    PTOTMatmulMXBiasToTMATMUL_MX_BIAS,
    PTOTGemvBiasToTGEMV_BIAS,
    PTOTGemvMXToTGEMV_MX,
    PTOTGemvMXAccToTGEMV_MX,
    PTOTGemvMXBiasToTGEMV_MX,
    PTOBarrierToEmitC
  >(typeConverter, ctx);

  patterns.add<CallToEmitC, ReturnToEmitC>(typeConverter, ctx);

  populateSCFToEmitCConversionPatterns(patterns);
  // Keep CFG-style branches type-consistent when block argument types are
  // converted (e.g. after lowering scf.while to cf.br/cf.cond_br).
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct EmitPTOManualPass
    : public PassWrapper<EmitPTOManualPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitPTOManualPass)

  PTOArch targetArch;

  EmitPTOManualPass() : targetArch(PTOArch::A3) {}

  explicit EmitPTOManualPass(PTOArch arch) : targetArch(arch) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect, func::FuncDialect, arith::ArithDialect,
                    memref::MemRefDialect, affine::AffineDialect,
                    mlir::cf::ControlFlowDialect, mlir::pto::PTODialect>();
  }

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "DEBUG: Start PTOToEmitC Pass\n");
    MLIRContext *ctx = &getContext();
    ModuleOp mop = getOperation();

    if (failed(pto::validatePTOEntryFunctions(mop)))
      return signalPassFailure();
    pto::annotatePTOEntryFunctions(mop);

    // A3 requires explicit FFTS base setup for inter-core sync ops.
    if (targetArch == PTOArch::A3) {
      bool hasMissingSetFFTs = false;
      for (auto func : mop.getOps<func::FuncOp>()) {
        if (!hasInterCoreSyncOp(func))
          continue;
        if (hasSetFFTsOp(func))
          continue;
        hasMissingSetFFTs = true;
        func.emitError()
            << "A3 inter-core sync requires explicit `pto.set_ffts` in the "
               "same function when using `pto.sync.set`/`pto.sync.wait`";
      }
      if (hasMissingSetFFTs)
        return signalPassFailure();
    }

        bool needsEventIdArrayHelper = false;
        bool needsTRandomHelper = false;
        bool needsGlobalTensorDataHelper = false;
        bool needsCommInclude = false;
        mop.walk([&](Operation *op) {
          if (isa<mlir::pto::DeclareEventIdArrayOp>(op))
            needsEventIdArrayHelper = true;
          if (isa<mlir::pto::TRandomOp>(op))
            needsTRandomHelper = true;
          if (isa<mlir::pto::PartitionViewOp>(op))
            needsGlobalTensorDataHelper = true;
          if (isa<mlir::pto::BuildAsyncSessionOp, mlir::pto::TPutAsyncOp,
                  mlir::pto::TGetAsyncOp, mlir::pto::TPrefetchAsyncOp,
                  mlir::pto::WaitAsyncEventOp, mlir::pto::TestAsyncEventOp,
                  mlir::pto::TPutOp,
                  mlir::pto::TGetOp, mlir::pto::TNotifyOp, mlir::pto::TWaitOp,
                  mlir::pto::TTestOp, mlir::pto::TBroadcastOp,
                  mlir::pto::CommTGatherOp, mlir::pto::CommTScatterOp,
                  mlir::pto::TReduceOp>(op))
            needsCommInclude = true;
        });

		    // 1. 插入头文件
	    auto loc = mop->getLoc();
	    OpBuilder builder(ctx);
	    builder.setInsertionPointToStart(mop.getBody());
	    builder.create<emitc::IncludeOp>(
	        loc, "pto/pto-inst.hpp", /*is_standard_include=*/false);
        if (needsCommInclude) {
	      builder.create<emitc::VerbatimOp>(
	          loc, builder.getStringAttr(R"cpp(
#ifndef PIPE_FIX
#define PIPE_FIX PIPE_M
#endif
)cpp"));
	      builder.create<emitc::IncludeOp>(
	          loc, "pto/comm/pto_comm_inst.hpp", /*is_standard_include=*/false);
        }
	    builder.create<emitc::VerbatimOp>(
	        loc, builder.getStringAttr("using namespace pto;"));
        if (needsGlobalTensorDataHelper) {
	      builder.create<emitc::VerbatimOp>(
	          loc, builder.getStringAttr(R"cpp(
template <typename Tensor>
static AICORE inline auto PTOAS__GLOBAL_TENSOR_DATA(Tensor &tensor)
    -> decltype(tensor.data()) {
  return tensor.data();
}
)cpp"));
        }
        if (needsEventIdArrayHelper) {
	      builder.create<emitc::VerbatimOp>(
	          loc, builder.getStringAttr(R"cpp(
template <int N>
struct PTOAS_EventIdArray {
  static_assert(N > 0, "PTOAS_EventIdArray requires a positive static size");
  int32_t data[N] = {};

  AICORE inline int32_t &operator[](int32_t idx) { return data[idx]; }
  AICORE inline const int32_t &operator[](int32_t idx) const { return data[idx]; }
};
)cpp"));
        }
        if (needsTRandomHelper) {
	      builder.create<emitc::VerbatimOp>(
	          loc, builder.getStringAttr(R"cpp(
template <uint16_t Rounds, typename DstTile>
static AICORE inline void PTOAS__TRANDOM(
    DstTile &dst, uint32_t key0, uint32_t key1, uint32_t counter0,
    uint32_t counter1, uint32_t counter2, uint32_t counter3) {
  TRandomKey key = {key0, key1};
  TRandomCounter counter = {counter0, counter1, counter2, counter3};
  TRANDOM<Rounds>(dst, key, counter);
}
)cpp"));
        }
	    builder.create<emitc::VerbatimOp>(
	        loc, builder.getStringAttr(R"cpp(
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
)cpp"));
	    // Only inject the bitcast helper when we actually lower ops that need it
	    // (e.g. arith.bitcast or arith.maximumf/minimumf tie-breaking on zeros).
	    bool needsBitcastHelper = false;
	    mop.walk([&](Operation *op) {
	      if (isa<arith::BitcastOp, arith::MaximumFOp, arith::MinimumFOp>(op)) {
	        needsBitcastHelper = true;
	        return WalkResult::interrupt();
	      }
	      return WalkResult::advance();
	    });
	    if (needsBitcastHelper) {
	      builder.create<emitc::VerbatimOp>(
	          loc, builder.getStringAttr(R"cpp(
		template <typename To, typename From>
		static inline To ptoas_bitcast(From from) {
		  static_assert(sizeof(To) == sizeof(From), "ptoas_bitcast: size mismatch");
		  To to;
		  __builtin_memcpy(&to, &from, sizeof(To));
		  return to;
		}
		)cpp"));
	    }

	    // 1.5 Pre-lower SCF constructs not handled by SCFToEmitC.
	    {
	      // scf.while / scf.index_switch are lowered via CFG blocks. This is not
      // possible inside ops that require single-block regions (e.g. scf.for /
      // scf.if). If we see such nesting, lower the entire function to the
      // ControlFlow dialect first.
      bool needsAnySCFToCF = false;
      for (auto func : mop.getOps<func::FuncOp>()) {
        if (needsWholeFunctionSCFToCF(func)) {
          needsAnySCFToCF = true;
          break;
        }
      }
      if (needsAnySCFToCF) {
        RewritePatternSet scfToCfPatterns(ctx);
        populateSCFToControlFlowConversionPatterns(scfToCfPatterns);
        FrozenRewritePatternSet frozenSCFToCF(std::move(scfToCfPatterns));

        ConversionTarget scfToCfTarget(*ctx);
        // Only eliminate the single-block SCF constructs; we'll pre-lower
        // scf.while/index_switch/execute_region ourselves afterwards.
        scfToCfTarget.addIllegalOp<scf::ForallOp, scf::ForOp, scf::IfOp,
                                   scf::ParallelOp>();
        scfToCfTarget.markUnknownOpDynamicallyLegal(
            [](Operation *) { return true; });

        for (auto func : mop.getOps<func::FuncOp>()) {
          if (!needsWholeFunctionSCFToCF(func))
            continue;
          if (failed(applyPartialConversion(func, scfToCfTarget,
                                            frozenSCFToCF))) {
            func.emitError()
                << "failed to lower nested SCF to ControlFlow (SCFToCF)";
            return signalPassFailure();
          }
        }
      }

      RewritePatternSet scfLoweringPatterns(ctx);
      scfLoweringPatterns.add<SCFExecuteRegionInline, SCFExecuteRegionToCF,
                              SCFIndexSwitchToCF,
                              SCFWhileToCF, CFSwitchToCondBr>(ctx);
      (void)applyPatternsAndFoldGreedily(mop, std::move(scfLoweringPatterns));

      bool hasUnsupportedSCF = false;
      mop.walk([&](Operation *op) {
        if (isa<scf::ExecuteRegionOp, scf::IndexSwitchOp, scf::WhileOp>(op)) {
          hasUnsupportedSCF = true;
          op->emitError() << "Unsupported SCF op remained after pre-lowering";
          return WalkResult::interrupt();
        }
        if (isa<cf::SwitchOp>(op)) {
          hasUnsupportedSCF = true;
          op->emitError()
              << "Unsupported CF op remained after pre-lowering: cf.switch";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (hasUnsupportedSCF)
        return signalPassFailure();
    }

    PTOToEmitCTypeConverter typeConverter(ctx, targetArch);

    // 2. Pre-convert SCF structural op types (e.g. scf.if/scf.for results)
    // using the same type converter. This avoids creating emitc.variable with
    // unsupported types such as memref.
    {
      RewritePatternSet scfTypePatterns(ctx);
      ConversionTarget scfTypeTarget(*ctx);
      scf::populateSCFStructuralTypeConversionsAndLegality(
          typeConverter, scfTypePatterns, scfTypeTarget);
      scfTypeTarget.markUnknownOpDynamicallyLegal(
          [](Operation *) { return true; });

      if (failed(applyPartialConversion(mop, scfTypeTarget,
                                        std::move(scfTypePatterns)))) {
        mop.emitError("failed to reconcile SCF structural types");
        return signalPassFailure();
      }
    }

    annotateTNotifyMteDrain(mop);

    // 3. 配置转换目标
    ConversionTarget target(*ctx);

    target.addIllegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<pto::PTODialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<mlir::scf::SCFDialect>(); 
    
    // If we introduced CFG branches (e.g. from scf.while), make sure they are
    // updated to use legalized operand types.
    target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
        [&](Operation *op) {
          return isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                                  typeConverter);
        });

    // [关键] 允许 Cast 存在，最后统一清理
    target.addLegalOp<UnrealizedConversionCastOp>(); 

    target.addIllegalOp<func::ReturnOp>();
    target.addIllegalOp<func::FuncOp>(); 
    target.addIllegalOp<func::CallOp>();

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalOp<ModuleOp>();

    auto solver = std::make_unique<DataFlowSolver>();
    solver->load<dataflow::DeadCodeAnalysis>();
    solver->load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    RewritePatternSet patterns(ctx);
    populatePTOToEmitCPatterns(patterns, typeConverter, ctx, *solver, targetArch);

    // 4. 执行转换
    if (failed(applyPartialConversion(mop, target, std::move(patterns)))) {
      llvm::errs() << "Conversion FAILED! Rolling back executed.\n";
      return signalPassFailure();
    }

    // =========================================================================
    // 5. [终极清理] 
    // 顺序至关重要：
    // Step A: 先移除所有 Cast，让 Loop 的 Operand 类型变成底层类型 (如 int32)
    // Step B: 再根据新的 Operand 类型，修复 Loop IV 的类型
    // =========================================================================
    
    // --- Step A: 清理 UnrealizedConversionCastOp ---
    // Prefer dropping redundant/unused casts; otherwise lower to emitc.cast
    // so the C++ emitter can print it.
    auto isEmitCTileLikeType = [](Type ty) {
      auto opaqueTy = dyn_cast<emitc::OpaqueType>(ty);
      if (!opaqueTy)
        return false;
      StringRef value = opaqueTy.getValue();
      return value.contains("Tile<") || value.contains("ConvTile<");
    };

    llvm::SmallVector<UnrealizedConversionCastOp> castsToErase;
    bool castCleanupFailed = false;
    mop.walk([&](UnrealizedConversionCastOp cast) {
      if (castCleanupFailed)
        return;

      if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
        cast.emitError() << "unsupported unrealized_conversion_cast shape";
        castCleanupFailed = true;
        return;
      }

      Value input = cast.getOperand(0);
      Value output = cast.getResult(0);
      Type inTy = input.getType();
      Type outTy = output.getType();

      if (output.use_empty()) {
        castsToErase.push_back(cast);
        return;
      }

      if (inTy == outTy) {
        output.replaceAllUsesWith(input);
        castsToErase.push_back(cast);
        return;
      }

      // SCF/CFG type conversion can transiently materialize pointer->memref
      // bridge casts. At this stage, the producing value is already in the
      // lowered EmitC pointer form; keep it and drop the bridge cast.
      if (isEmitCPointerLikeType(inTy) && isa<BaseMemRefType>(outTy)) {
        output.replaceAllUsesWith(input);
        castsToErase.push_back(cast);
        return;
      }

      // SCF structural type conversion may leave a bridge from the converted
      // EmitC tile value back to the original pto.tile_buf type for PTO op
      // users. After PTO ops are lowered, the EmitC tile value is the value we
      // want to keep.
      if (isEmitCTileLikeType(inTy) && isa<pto::TileBufType>(outTy)) {
        output.replaceAllUsesWith(input);
        castsToErase.push_back(cast);
        return;
      }

      if (emitc::isSupportedEmitCType(inTy) && emitc::isSupportedEmitCType(outTy)) {
        OpBuilder builder(cast);
        auto c = builder.create<emitc::CastOp>(cast.getLoc(), outTy, input);
        output.replaceAllUsesWith(c.getResult());
        castsToErase.push_back(cast);
        return;
      }

      cast.emitError() << "cannot lower unrealized_conversion_cast(" << inTy
                       << " -> " << outTy << ") to emitc.cast";
      castCleanupFailed = true;
    });

    for (auto cast : castsToErase)
      cast.erase();

    if (castCleanupFailed)
      return signalPassFailure();

    // --- Step A2: Sink casts of emitc.variable "reads" to their use sites ---
    //
    // SCFToEmitC lowers scf.if/scf.for results via mutable `emitc.variable` and
    // `emitc.assign`. During type conversion, casts from the variable handle to
    // the converted type may be materialized right after the variable
    // declaration, effectively snapshotting the value *before* assignments. That
    // produces wrong C++ (use-before-init / stale reads).
    //
    // Fix by re-materializing the cast at each use site so it reads the variable
    // at the point of use.
    {
      SmallVector<emitc::CastOp> castOpsToSink;
      mop.walk([&](emitc::CastOp castOp) {
        if (castOp.getSource().getDefiningOp<emitc::VariableOp>())
          castOpsToSink.push_back(castOp);
      });

      for (emitc::CastOp castOp : castOpsToSink) {
        Value src = castOp.getSource();
        Type dstTy = castOp.getResult().getType();
        Value oldRes = castOp.getResult();

        // Replace each use with a freshly inserted cast right before the user.
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

    // --- Step B: 修复 Loop 归纳变量 (IV) ---
    // 此时 emitc.for 的 operand 已经是 int32 了，我们检查 IV 是否匹配，不匹配则修正
    mop.walk([&](emitc::ForOp forOp) {
       Type boundTy = forOp.getLowerBound().getType(); 
       BlockArgument iv = forOp.getBody()->getArgument(0); 
       
       if (iv.getType() != boundTy) {
         iv.setType(boundTy); // 强制将 IV 类型 (index) 修改为与边界一致 (int32)
       }
    });
    
    // --- Step C: 消除冗余 Tile 变量 (Dead Code Elimination) [新增] ---
    // 逻辑：如果一个 emitc.variable 没有被读取（use_empty），
    // 那么它自己，以及给它赋值的 TASSIGN 都可以删除。
    // 注意：TASSIGN(v15, v9) 会把 v15 作为 Operand 0 使用，所以 v15 不是严格的 use_empty。
    // 我们需要检查：v15 是否除了 TASSIGN 之外没有其他 User。

    llvm::SmallVector<emitc::VariableOp> deadVars;
    mop.walk([&](emitc::VariableOp varOp) {
        // 检查该变量的所有 User
        bool isRead = false;
        for (Operation* user : varOp.getResult().getUsers()) {
            // 如果 User 是 TASSIGN 且变量是第0个参数(dst)，不算"读取"
            if (auto call = dyn_cast<emitc::CallOpaqueOp>(user)) {
                if (call.getCallee() == "TASSIGN" && call.getOperand(0) == varOp.getResult()) {
                    continue; // 这是一个赋值操作，不算有效使用
                }
            }
            // 如果还有其他用途（如 TLOAD, TMOV, TMATMUL），则该变量有用
            isRead = true;
            break;
        }

        if (!isRead) {
            deadVars.push_back(varOp);
        }
    });

    for (auto varOp : deadVars) {
        // 1. 先删除所有使用该变量的 TASSIGN
        llvm::SmallVector<Operation*> usersToErase;
        for (Operation* user : varOp.getResult().getUsers()) {
             // 我们上面已经确认过，剩下的 user 只能是 TASSIGN
             usersToErase.push_back(user);
        }
        for (auto u : usersToErase) u->erase();

        // 2. 删除变量定义本身
        varOp.erase();
    }

    llvm::SmallVector<emitc::ConstantOp> deadConsts;
    mop.walk([&](emitc::ConstantOp constOp) {
      if (constOp.getResult().use_empty())
        deadConsts.push_back(constOp);
    });
    for (auto constOp : deadConsts)
      constOp.erase();

    // =========================================================================
  }
  };
} // namespace

std::unique_ptr<Pass> mlir::pto::createEmitPTOManualPass() {
  return std::make_unique<EmitPTOManualPass>();
}

std::unique_ptr<Pass> mlir::pto::createEmitPTOManualPass(PTOArch arch) {
  return std::make_unique<EmitPTOManualPass>(arch);
}

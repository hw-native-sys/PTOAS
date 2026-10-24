// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOA5NORMALIZETMOV
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr size_t kTileRank2D = 2;
constexpr size_t kFirstTileDim = 0;
constexpr size_t kSecondTileDim = 1;
constexpr unsigned kRiskyOpReserveSize = 8;
constexpr unsigned kTMovOperandReserveSize = 4;

static bool isVecTileType(pto::TileBufType type) {
  auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(type.getMemorySpace());
  return asAttr && asAttr.getAddressSpace() == pto::AddressSpace::VEC;
}

static bool isColMajorNoneBox(pto::TileBufType type) {
  return type.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::ColMajor) &&
         type.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::NoneBox);
}

static bool isA5RiskyVecVecColMajorTMov(pto::TMovOp op) {
  if (pto::classifyTMovForm(op.getFp()) == pto::TMovForm::XToZz) {
    return false;
  }
  auto srcTb = dyn_cast<pto::TileBufType>(op.getSrc().getType());
  auto dstTb = dyn_cast<pto::TileBufType>(op.getDst().getType());
  if (!srcTb || !dstTb) {
    return false;
  }
  if (!isVecTileType(srcTb) || !isVecTileType(dstTb)) {
    return false;
  }
  return isColMajorNoneBox(srcTb) && isColMajorNoneBox(dstTb);
}

static std::optional<pto::AddressSpace> getAddressSpaceFromValueType(Type type) {
  if (auto tb = dyn_cast<pto::TileBufType>(type)) {
    if (auto as =
            dyn_cast_or_null<pto::AddressSpaceAttr>(tb.getMemorySpace())) {
      return as.getAddressSpace();
    }
    return std::nullopt;
  }
  if (auto mr = dyn_cast<MemRefType>(type)) {
    if (auto ms = mr.getMemorySpace()) {
      if (auto as = dyn_cast<pto::AddressSpaceAttr>(ms)) {
        return as.getAddressSpace();
      }
    }
  }
  return std::nullopt;
}

static bool isA5ScaleTileTMov(pto::TMovOp op) {
  auto srcAS = getAddressSpaceFromValueType(op.getSrc().getType());
  auto dstAS = getAddressSpaceFromValueType(op.getDst().getType());
  return srcAS && dstAS && *srcAS == pto::AddressSpace::MAT &&
         *dstAS == pto::AddressSpace::SCALING;
}

static Operation *findInterveningUseOfDst(Operation *start, const Operation *end,
                                        Value dst) {
  for (Operation *cursor = start->getNextNode(); cursor && cursor != end;
       cursor = cursor->getNextNode()) {
    Operation *use = nullptr;
    cursor->walk([&](Operation *nested) {
      if (llvm::is_contained(nested->getOperands(), dst)) {
        use = nested;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (use) {
      return use;
    }
  }
  return nullptr;
}

static bool isA5ScaleTileTExtract(pto::TExtractOp op) {
  auto srcAS = getAddressSpaceFromValueType(op.getSrc().getType());
  auto dstAS = getAddressSpaceFromValueType(op.getDst().getType());
  return srcAS && dstAS && *srcAS == pto::AddressSpace::MAT &&
         *dstAS == pto::AddressSpace::SCALING && !op.getFp() &&
         !op.getPreQuantScalar() &&
         op.getReluPreMode() == pto::ReluPreMode::NoRelu &&
         !op.getAccToVecModeAttr();
}

static Operation *findMatchingScaleTileProducer(pto::TGetScaleAddrOp op) {
  Value dst = op.getDst();
  for (Operation *cursor = op->getPrevNode(); cursor;
       cursor = cursor->getPrevNode()) {
    if (auto mov = dyn_cast<pto::TMovOp>(cursor)) {
      if (mov.getDst() == dst && isA5ScaleTileTMov(mov)) {
        return cursor;
      }
    } else if (auto extract = dyn_cast<pto::TExtractOp>(cursor)) {
      if (extract.getDst() == dst && isA5ScaleTileTExtract(extract)) {
        return cursor;
      }
    }
  }
  return nullptr;
}

static bool isScaleAddrScalarType(Type type) {
  return isa<IntegerType, IndexType, FloatType>(type);
}

static bool isSameBlockTileAllocation(Value value, Block *block) {
  auto alloc = value.getDefiningOp<pto::AllocTileOp>();
  return alloc && alloc->getBlock() == block;
}

static bool preservesScaleAddrBinding(Operation *op, Value dst) {
  if (op->getNumRegions() || op->getNumSuccessors()) {
    return false;
  }
  if (auto binding = dyn_cast<pto::TGetScaleAddrOp>(op)) {
    // Distinct allocations are distinct mutable handles, even if their data
    // storage addresses coincide. Do not infer this for aliases/block arguments.
    return binding.getDst() != dst &&
           isSameBlockTileAllocation(binding.getDst(), op->getBlock());
  }
  if (auto mov = dyn_cast<pto::TMovOp>(op)) {
    // These copies write tile data, not the tile handle's address. Keep the
    // unrelated quantizing forms outside this narrowly scoped proof.
    return pto::classifyTMovForm(mov.getFp()) == pto::TMovForm::NoTileAux;
  }
  if (auto extract = dyn_cast<pto::TExtractOp>(op)) {
    // A base Mat->Scaling extraction writes data through the bound handle.
    return isA5ScaleTileTExtract(extract);
  }
  // Explicitly known fixed-handle data operations and fresh descriptors only.
  // In particular, tassign is Pure in ODS but mutates the emitted tile handle:
  // accepting arbitrary pure tile operations would make this proof unsound.
  if (isa<pto::AllocTileOp, pto::MakeTensorViewOp, pto::PartitionViewOp,
          pto::TMatmulMxOp, pto::TMatmulMxAccOp, pto::TStoreOp>(op)) {
    return true;
  }
  return llvm::all_of(op->getOperandTypes(), isScaleAddrScalarType) &&
         llvm::all_of(op->getResultTypes(), isScaleAddrScalarType) &&
         isPure(op);
}

static bool hasEstablishedScaleAddrBinding(pto::TGetScaleAddrOp op,
                                          Operation *matchingProducer) {
  // Only reason about independently allocated handles in this block. Do not
  // attempt to prove cross-block/region or alias equivalence.
  Block *block = op->getBlock();
  if (!isSameBlockTileAllocation(op.getSrc(), block) ||
      !isSameBlockTileAllocation(op.getDst(), block)) {
    return false;
  }
  for (Operation *cursor = op->getPrevNode(); cursor;
       cursor = cursor->getPrevNode()) {
    if (auto binding = dyn_cast<pto::TGetScaleAddrOp>(cursor)) {
      if (binding.getDst() == op.getDst()) {
        if (binding.getSrc() != op.getSrc()) {
          return false;
        }
        // Equal repeated bindings preserve the state, but only an earlier
        // binding that dominates the copy can establish the required proof.
        if (binding->isBeforeInBlock(matchingProducer)) {
          return true;
        }
        continue;
      }
    }
    if (!preservesScaleAddrBinding(cursor, op.getDst())) {
      return false;
    }
  }
  return false;
}

static bool canHoistScaleAddrDependency(Operation *op) {
  if (op->getNumRegions() || op->getNumSuccessors()) {
    return false;
  }
  // alloc_tile creates a logical handle, not a data transfer. Its address and
  // dynamic valid-shape operands are checked recursively before moving it.
  if (isa<pto::AllocTileOp>(op)) {
    return true;
  }
  return llvm::all_of(op->getOperandTypes(), isScaleAddrScalarType) &&
         llvm::all_of(op->getResultTypes(), isScaleAddrScalarType) &&
         isPure(op);
}

// Collect the complete dependency slice without modifying IR. In particular,
// a rejected later operand must not leave an earlier allocation half-hoisted.
static LogicalResult collectScaleAddrDependencies(
    Value value, Operation *anchor, Operation *scaleAddr,
    DominanceInfo &dominance, llvm::SmallPtrSetImpl<Operation *> &collected,
    SmallVectorImpl<Operation *> &dependencies, Operation *&blockingDef) {
  if (dominance.dominates(value, anchor)) {
    return success();
  }
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != anchor->getBlock() ||
      !anchor->isBeforeInBlock(def) || !def->isBeforeInBlock(scaleAddr) ||
      !canHoistScaleAddrDependency(def)) {
    blockingDef = def;
    return failure();
  }
  if (collected.contains(def)) {
    return success();
  }
  for (Value operand : def->getOperands()) {
    if (failed(collectScaleAddrDependencies(operand, anchor, scaleAddr,
                                           dominance, collected, dependencies,
                                           blockingDef))) {
      return failure();
    }
  }
  collected.insert(def);
  dependencies.push_back(def);
  return success();
}

static LogicalResult hoistScaleAddr(pto::TGetScaleAddrOp op,
                                   Operation *matchingProducer,
                                   DominanceInfo &dominance) {
  StringRef producerName = isa<pto::TMovOp>(matchingProducer) ? "TMOV" : "TEXTRACT";
  // The binding changes the destination address, including for uses captured
  // by nested regions. Neither crossing those uses nor silently keeping the
  // binding after the copy establishes safe backend semantics.
  if (Operation *use = findInterveningUseOfDst(matchingProducer, op, op.getDst())) {
    auto diagnostic = op.emitOpError();
    diagnostic << "cannot safely establish scaling address before matching "
               << producerName << ": intervening destination use";
    diagnostic.attachNote(use->getLoc())
        << "intervening destination use: " << use->getName();
    diagnostic.attachNote(matchingProducer->getLoc())
        << "matching " << producerName << " is here";
    return failure();
  }
  llvm::SmallPtrSet<Operation *, kRiskyOpReserveSize> collected;
  SmallVector<Operation *, kRiskyOpReserveSize> dependencies;
  Operation *blockingDef = nullptr;
  for (Value operand : op->getOperands()) {
    if (failed(collectScaleAddrDependencies(
            operand, matchingProducer, op, dominance, collected, dependencies,
            blockingDef))) {
      auto diagnostic = op.emitOpError();
      diagnostic << "cannot safely establish scaling address before matching "
                 << producerName << ": operand dependency cannot be hoisted";
      if (blockingDef) {
        diagnostic.attachNote(blockingDef->getLoc())
            << "blocking dependency: " << blockingDef->getName();
      }
      diagnostic.attachNote(matchingProducer->getLoc())
          << "matching " << producerName << " is here";
      return failure();
    }
  }
  for (Operation *dependency : dependencies) {
    dependency->moveBefore(matchingProducer);
  }
  op->moveBefore(matchingProducer);
  return success();
}

// A constant binding is removable only when both handles remain local and
// every use accesses data through a fixed handle. Aliases, tassign (even
// though Pure), calls and captured region uses retain the mutable binding:
// memory effects alone cannot establish handle stability.
static bool isFixedMxHandleUse(OpOperand &use, bool isScale) {
  Operation *user = use.getOwner();
  if (user->getNumRegions() || user->getNumSuccessors() ||
      user->getNumResults()) {
    return false;
  }
  Value value = use.get();
  if (auto mov = dyn_cast<pto::TMovOp>(user)) {
    return pto::classifyTMovForm(mov.getFp()) == pto::TMovForm::NoTileAux &&
           (!isScale || (mov.getDst() == value && isA5ScaleTileTMov(mov)));
  }
  if (auto extract = dyn_cast<pto::TExtractOp>(user)) {
    if (isScale) {
      return extract.getDst() == value && isA5ScaleTileTExtract(extract);
    }
    // A5 MX K slices use TEXTRACT rather than TMOV to fill a fixed L0
    // handle. Only the destination payload is written: a source use, alias,
    // quantizing form or different memory path cannot establish this proof.
    // The TEXTRACT verifier owns layout/shape legality, including RHS layout
    // conversion; neither changes the allocated destination handle address.
    auto sourceType = dyn_cast<pto::TileBufType>(extract.getSrc().getType());
    auto sourceAS = getAddressSpaceFromValueType(extract.getSrc().getType());
    auto destAS = getAddressSpaceFromValueType(extract.getDst().getType());
    return extract.getDst() == value && sourceType && sourceAS && destAS &&
           *sourceAS == pto::AddressSpace::MAT &&
           (*destAS == pto::AddressSpace::LEFT ||
            *destAS == pto::AddressSpace::RIGHT) &&
           isa<Float8E4M3FNType>(sourceType.getElementType()) &&
           !extract.getFp() && !extract.getPreQuantScalar() &&
           extract.getReluPreMode() == pto::ReluPreMode::NoRelu &&
           !extract.getAccToVecModeAttr();
  }
  if (auto matmul = dyn_cast<pto::TMatmulMxOp>(user)) {
    return isScale ? (matmul.getAScale() == value ||
                      matmul.getBScale() == value)
                   : (matmul.getA() == value || matmul.getB() == value);
  }
  if (auto matmul = dyn_cast<pto::TMatmulMxAccOp>(user)) {
    return isScale ? (matmul.getAScale() == value ||
                      matmul.getBScale() == value)
                   : (matmul.getA() == value || matmul.getB() == value);
  }
  return false;
}

static void foldConstantScaleAddr(pto::TGetScaleAddrOp op,
                                  IRRewriter &rewriter) {
  Block *block = op->getBlock();
  auto data = op.getSrc().getDefiningOp<pto::AllocTileOp>();
  auto scale = op.getDst().getDefiningOp<pto::AllocTileOp>();
  if (!data || !scale || !data.getAddr() || data->getBlock() != block ||
      scale->getBlock() != block) {
    return;
  }
  auto dataType = dyn_cast<pto::TileBufType>(op.getSrc().getType());
  auto scaleType = dyn_cast<pto::TileBufType>(op.getDst().getType());
  auto dataAS = getAddressSpaceFromValueType(op.getSrc().getType());
  auto scaleAS = getAddressSpaceFromValueType(op.getDst().getType());
  if (!dataType || !scaleType || !dataAS || !scaleAS ||
      (*dataAS != pto::AddressSpace::LEFT &&
       *dataAS != pto::AddressSpace::RIGHT) ||
      *scaleAS != pto::AddressSpace::SCALING ||
      !isa<Float8E4M3FNType>(dataType.getElementType()) ||
      !isa<pto::F8E8M0Type>(scaleType.getElementType())) {
    return;
  }
  auto constant = data.getAddr().getDefiningOp<arith::ConstantOp>();
  auto address = constant ? dyn_cast<IntegerAttr>(constant.getValue())
                          : IntegerAttr();
  if (!address || address.getValue().isNegative() ||
      address.getValue().getBitWidth() != 64) {
    return;
  }
  uint64_t dataAddress = address.getValue().getZExtValue();
  if (dataAddress % 16 != 0) {
    return;
  }

  // Exactly one binding, no earlier scale access and no aliases. Direct
  // same-block users after the binding are dominated by it, also in loops.
  for (OpOperand &use : op.getDst().getUses()) {
    Operation *user = use.getOwner();
    if (user == op.getOperation()) {
      continue;
    }
    if (user->getBlock() != block || !op->isBeforeInBlock(user) ||
        !isFixedMxHandleUse(use, /*isScale=*/true)) {
      return;
    }
  }
  for (OpOperand &use : op.getSrc().getUses()) {
    Operation *user = use.getOwner();
    if (user == op.getOperation()) {
      continue;
    }
    if (user->getBlock() != block ||
        !isFixedMxHandleUse(use, /*isScale=*/false)) {
      return;
    }
  }

  // GetScaleAddr is the data address shifted by four, not a payload read.
  // Use a fresh constant: the old allocation address may be shared by other
  // handles. Real data/scale dependencies retain the final physical address.
  rewriter.setInsertionPoint(scale);
  Value scaleAddress = rewriter.create<arith::ConstantIntOp>(
      op.getLoc(), dataAddress >> 4, 64);
  scale.getAddrMutable().assign(scaleAddress);
  rewriter.eraseOp(op);
}

template <typename CfgT>
static auto buildRowMajorConfigImpl(int, MLIRContext *ctx,
                                    pto::BLayoutAttr rowMajor, CfgT cfg)
    -> decltype(pto::TileBufConfigAttr::get(ctx, rowMajor, cfg.getSLayout(),
                                            cfg.getSFractalSize(), cfg.getPad(),
                                            cfg.getCompactMode())) {
  return pto::TileBufConfigAttr::get(ctx, rowMajor, cfg.getSLayout(),
                                     cfg.getSFractalSize(), cfg.getPad(),
                                     cfg.getCompactMode());
}

template <typename CfgT>
static auto buildRowMajorConfigImpl(long, MLIRContext *ctx,
                                    pto::BLayoutAttr rowMajor, CfgT cfg)
    -> decltype(pto::TileBufConfigAttr::get(ctx, rowMajor, cfg.getSLayout(),
                                            cfg.getSFractalSize(),
                                            cfg.getPad())) {
  return pto::TileBufConfigAttr::get(ctx, rowMajor, cfg.getSLayout(),
                                     cfg.getSFractalSize(), cfg.getPad());
}

static pto::TileBufConfigAttr buildRowMajorConfig(MLIRContext *ctx,
                                                  pto::TileBufConfigAttr cfg) {
  auto rowMajor = pto::BLayoutAttr::get(ctx, pto::BLayout::RowMajor);
  return buildRowMajorConfigImpl(0, ctx, rowMajor, cfg);
}

static FailureOr<pto::TileBufType>
buildRowMajorReinterpretType(MLIRContext *ctx, pto::TileBufType srcType) {
  ArrayRef<int64_t> shape = srcType.getShape();
  if (shape.size() != kTileRank2D) {
    return failure();
  }
  if (shape[kFirstTileDim] == ShapedType::kDynamic ||
      shape[kSecondTileDim] == ShapedType::kDynamic) {
    return failure();
  }

  SmallVector<int64_t, kTileRank2D> swappedShape{shape[kSecondTileDim],
                                                 shape[kFirstTileDim]};

  SmallVector<int64_t, kTileRank2D> swappedValid;
  ArrayRef<int64_t> validShape = srcType.getValidShape();
  if (validShape.empty()) {
    swappedValid = swappedShape;
  } else if (validShape.size() == kTileRank2D) {
    swappedValid.assign({validShape[kSecondTileDim],
                         validShape[kFirstTileDim]});
  } else {
    return failure();
  }

  auto cfg = srcType.getConfigAttr();
  if (!cfg) {
    cfg = pto::TileBufConfigAttr::getDefault(ctx);
  }
  auto newCfg = buildRowMajorConfig(ctx, cfg);

  return pto::TileBufType::get(ctx, swappedShape, srcType.getElementType(),
                               srcType.getMemorySpace(), swappedValid, newCfg);
}

static void setSwappedDynamicValidShapeIfNeeded(
    IRRewriter &rewriter, Location loc, Value sourceTile, Value reshapedTile,
    pto::TileBufType reshapedType) {
  if (!reshapedType.hasDynamicValid()) {
    return;
  }

  auto validShape = rewriter.create<pto::GetValidShapeOp>(loc, sourceTile);
  rewriter.create<pto::SetValidShapeOp>(
      loc, reshapedTile, validShape.getValidCol(), validShape.getValidRow());
}

struct PTOA5NormalizeTMovPass
    : public mlir::pto::impl::PTOA5NormalizeTMovBase<PTOA5NormalizeTMovPass> {
  void runOnOperation() override;

private:
  LogicalResult hoistScaleAddrOpsBeforeProducer(func::FuncOp func) const;
  LogicalResult normalizeRiskyVecVecTMov(func::FuncOp func) const;
  LogicalResult rewriteRiskyVecVecTMov(IRRewriter &rewriter, pto::TMovOp op) const;
  void verifyNoResidualRiskyTMov(func::FuncOp func);
};

LogicalResult PTOA5NormalizeTMovPass::hoistScaleAddrOpsBeforeProducer(
    func::FuncOp func) const {
  SmallVector<pto::TGetScaleAddrOp, kRiskyOpReserveSize> scaleAddrOps;
  DominanceInfo dominance(func);
  func.walk([&](pto::TGetScaleAddrOp op) { scaleAddrOps.push_back(op); });
  for (pto::TGetScaleAddrOp op : scaleAddrOps) {
    auto matchingProducer = findMatchingScaleTileProducer(op);
    if (!matchingProducer) {
      continue;
    }
    if (hasEstablishedScaleAddrBinding(op, matchingProducer)) {
      // Preserve the repeated binding's position; no reader is crossed and no
      // operation or dependency is erased or moved on this path.
      continue;
    }
    if (failed(hoistScaleAddr(op, matchingProducer, dominance))) {
      return failure();
    }
  }
  return success();
}

LogicalResult PTOA5NormalizeTMovPass::rewriteRiskyVecVecTMov(
    IRRewriter &rewriter, pto::TMovOp op) const {
  auto srcTb = cast<pto::TileBufType>(op.getSrc().getType());
  auto dstTb = cast<pto::TileBufType>(op.getDst().getType());

  FailureOr<pto::TileBufType> srcRowTy =
      buildRowMajorReinterpretType(op.getContext(), srcTb);
  FailureOr<pto::TileBufType> dstRowTy =
      buildRowMajorReinterpretType(op.getContext(), dstTb);
  if (failed(srcRowTy) || failed(dstRowTy)) {
    op.emitOpError(
        "cannot normalize A5 vec->vec col_major TMOV: requires static 2D "
        "tile_buf shape/valid_shape for treshape reinterpret");
    return failure();
  }

  rewriter.setInsertionPoint(op);
  auto srcRow =
      rewriter.create<pto::TReshapeOp>(op.getLoc(), *srcRowTy, op.getSrc());
  auto dstRow =
      rewriter.create<pto::TReshapeOp>(op.getLoc(), *dstRowTy, op.getDst());
  setSwappedDynamicValidShapeIfNeeded(
      rewriter, op.getLoc(), op.getSrc(), srcRow.getResult(), *srcRowTy);
  setSwappedDynamicValidShapeIfNeeded(
      rewriter, op.getLoc(), op.getDst(), dstRow.getResult(), *dstRowTy);
  SmallVector<Value, kTMovOperandReserveSize> newOperands(
      op->operand_begin(), op->operand_end());
  if (newOperands.size() < kTileRank2D) {
    op.emitOpError("unexpected operand count while normalizing TMOV");
    return failure();
  }
  newOperands[kFirstTileDim] = srcRow.getResult();
  newOperands[kSecondTileDim] = dstRow.getResult();

  OperationState state(op.getLoc(), pto::TMovOp::getOperationName());
  state.addOperands(newOperands);
  state.addTypes(op->getResultTypes());
  state.addAttributes(op->getAttrs());
  auto *created = rewriter.create(state);
  auto newTmov = cast<pto::TMovOp>(created);
  (void)newTmov;
  rewriter.eraseOp(op);
  return success();
}

LogicalResult PTOA5NormalizeTMovPass::normalizeRiskyVecVecTMov(
    func::FuncOp func) const {
  SmallVector<pto::TMovOp, kRiskyOpReserveSize> riskyOps;
  func.walk([&](pto::TMovOp op) {
    if (isA5RiskyVecVecColMajorTMov(op)) {
      riskyOps.push_back(op);
    }
  });

  IRRewriter rewriter(func.getContext());
  for (pto::TMovOp op : riskyOps) {
    if (failed(rewriteRiskyVecVecTMov(rewriter, op))) {
      return failure();
    }
  }
  return success();
}

void PTOA5NormalizeTMovPass::verifyNoResidualRiskyTMov(func::FuncOp func) {
  bool hasResidualRisk = false;
  func.walk([&](pto::TMovOp op) {
    if (!isA5RiskyVecVecColMajorTMov(op)) {
      return WalkResult::advance();
    }
    op.emitOpError(
        "A5 vec->vec TMOV on col_major/none_box tile is unsupported; "
        "expected normalization to row_major via pto.treshape");
    hasResidualRisk = true;
    return WalkResult::interrupt();
  });
  if (hasResidualRisk) {
    signalPassFailure();
  }
}

void PTOA5NormalizeTMovPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!isTargetArchA5(func.getOperation())) {
    return;
  }

  if (failed(hoistScaleAddrOpsBeforeProducer(func))) {
    signalPassFailure();
    return;
  }

  if (failed(normalizeRiskyVecVecTMov(func))) {
    signalPassFailure();
    return;
  }

  // Normalize all producers, retaining existing rejection and atomic
  // dependency checks, before considering removal of any binding.
  SmallVector<pto::TGetScaleAddrOp, kRiskyOpReserveSize> scaleAddrOps;
  func.walk([&](pto::TGetScaleAddrOp op) { scaleAddrOps.push_back(op); });
  IRRewriter rewriter(func.getContext());
  for (pto::TGetScaleAddrOp op : scaleAddrOps) {
    foldConstantScaleAddr(op, rewriter);
  }

  verifyNoResidualRiskyTMov(func);
}

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOA5NormalizeTMovPass() {
  return std::make_unique<PTOA5NormalizeTMovPass>();
}

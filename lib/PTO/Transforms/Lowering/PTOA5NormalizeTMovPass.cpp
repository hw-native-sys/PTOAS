// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
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

static bool isMatToScale(Value src, Value dst)
{
    auto srcAS = getAddressSpaceFromValueType(src.getType());
    auto dstAS = getAddressSpaceFromValueType(dst.getType());
    return srcAS && dstAS && *srcAS == pto::AddressSpace::MAT && *dstAS == pto::AddressSpace::SCALING;
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

static bool isBaseExtract(pto::TExtractOp op)
{
    return !op.getFp() && !op.getPreQuantScalar() && !op.getAccToVecModeAttr() &&
           op.getReluPreMode() == pto::ReluPreMode::NoRelu;
}

static bool isScaleTileCopy(Operation* op, Value dst)
{
    if (auto mov = dyn_cast<pto::TMovOp>(op)) {
        return mov.getDst() == dst && isMatToScale(mov.getSrc(), dst);
    }
    if (auto extract = dyn_cast<pto::TExtractOp>(op)) {
        return extract.getDst() == dst && isBaseExtract(extract) && isMatToScale(extract.getSrc(), dst);
    }
    return false;
}

static Operation* findMatchingScaleTileCopy(pto::TGetScaleAddrOp op)
{
    for (Operation* cursor = op->getPrevNode(); cursor; cursor = cursor->getPrevNode()) {
        if (isScaleTileCopy(cursor, op.getDst())) {
            return cursor;
        }
    }
    return nullptr;
}

static bool mayBindScaleExtract(pto::TGetScaleAddrOp op)
{
    const Value dst = op.getDst();
    Block* block = op->getBlock();
    for (Operation* user : dst.getUsers()) {
        const bool isMatchingExtract = isa<pto::TExtractOp>(user) && isScaleTileCopy(user, dst);
        if (!isMatchingExtract) {
            continue;
        }
        // A nested extract runs under its enclosing operation. An extract in
        // another block may also execute after this binding.
        Operation* ancestor = block->findAncestorOpInBlock(*user);
        const bool mayFollow = !ancestor || op->isBeforeInBlock(ancestor);
        if (mayFollow) {
            return true;
        }
    }
    return false;
}

static bool isScaleAddrScalarType(Type type) {
  return isa<IntegerType, IndexType, FloatType>(type);
}

static bool isSameBlockTileAllocation(Value value, const Block *block) {
  auto alloc = value.getDefiningOp<pto::AllocTileOp>();
  return alloc && alloc->getBlock() == block;
}

static bool preservesScaleAddrBinding(Operation* op, Value dst)
{
    const bool hasControlFlow = op->getNumRegions() != 0 || op->getNumSuccessors() != 0;
    if (hasControlFlow) {
        return false;
    }
    if (auto binding = dyn_cast<pto::TGetScaleAddrOp>(op)) {
        // Distinct allocations are distinct mutable handles, even if their data
        // storage addresses coincide. Do not infer this for aliases/block arguments.
        return binding.getDst() != dst && isSameBlockTileAllocation(binding.getDst(), op->getBlock());
    }
    if (auto mov = dyn_cast<pto::TMovOp>(op)) {
        // These copies write tile data, not the tile handle's address. Keep the
        // unrelated quantizing forms outside this narrowly scoped proof.
        return pto::classifyTMovForm(mov.getFp()) == pto::TMovForm::NoTileAux;
    }
    if (auto extract = dyn_cast<pto::TExtractOp>(op)) {
        return isBaseExtract(extract);
    }
    // Synchronization orders the pipes without rebinding tile addresses.
    if (isa<pto::SetFlagOp, pto::WaitFlagOp, pto::SetFlagDynOp, pto::WaitFlagDynOp, pto::BarrierOp, pto::GetBufOp,
        pto::RlsBufOp>(op)) {
        return true;
    }
    // Explicitly known fixed-handle data operations and fresh descriptors only.
    // In particular, tassign is Pure in ODS but mutates the emitted tile handle:
    // accepting arbitrary pure tile operations would make this proof unsound.
    if (isa<pto::AllocTileOp, pto::MakeTensorViewOp, pto::PartitionViewOp, pto::TMatmulMxOp, pto::TMatmulMxAccOp,
            pto::TStoreOp>(op)) {
        return true;
    }
    return llvm::all_of(op->getOperandTypes(), isScaleAddrScalarType) &&
           llvm::all_of(op->getResultTypes(), isScaleAddrScalarType) && isPure(op);
}

static bool hasEstablishedScaleAddrBinding(pto::TGetScaleAddrOp op, Operation* matchingCopy)
{
    // Only reason about independently allocated handles in this block. Do not
    // attempt to prove cross-block/region or alias equivalence.
    Block* block = op->getBlock();
    const bool independentAllocations =
        isSameBlockTileAllocation(op.getSrc(), block) && isSameBlockTileAllocation(op.getDst(), block);
    if (!independentAllocations) {
        return false;
    }
    for (Operation* cursor = op->getPrevNode(); cursor; cursor = cursor->getPrevNode()) {
        if (auto binding = dyn_cast<pto::TGetScaleAddrOp>(cursor)) {
            const bool matchesDst = binding.getDst() == op.getDst();
            if (matchesDst) {
                const bool matchesSrc = binding.getSrc() == op.getSrc();
                if (!matchesSrc) {
                    return false;
                }
                // Equal repeated bindings preserve the state, but only an earlier
                // binding that dominates the copy can establish the required proof.
                if (binding->isBeforeInBlock(matchingCopy)) {
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
  if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0) {
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

static bool canHoistScaleExtractBinding(pto::TGetScaleAddrOp op, Operation* copy)
{
    // An addressed allocation already lowers to TASSIGN. Captured handles
    // and results such as tassign may also carry an established address.
    auto alloc = op.getDst().getDefiningOp<pto::AllocTileOp>();
    const bool isUnboundLocalAllocation = alloc && alloc->getBlock() == op->getBlock() && !alloc.getAddr();
    if (!isUnboundLocalAllocation) {
        return false;
    }
    // A prior use can bind the handle or expose it through an alias.
    if (findInterveningUseOfDst(alloc, copy, op.getDst())) {
        return false;
    }
    // Moving the binding must preserve the source address as well as the
    // destination. Cross only fresh allocations and pure scalar computations.
    for (Operation* cursor = copy->getNextNode(); cursor != op; cursor = cursor->getNextNode()) {
        if (!canHoistScaleAddrDependency(cursor)) {
            return false;
        }
    }
    return true;
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

static LogicalResult hoistScaleAddr(pto::TGetScaleAddrOp op, Operation* matchingCopy, DominanceInfo& dominance)
{
    // A previous consumer's binding may already have been normalized directly
    // before this fill. The copy therefore used the requested address. Keep a
    // later identical binding at its original consumer: moving it back across
    // earlier uses is unnecessary, and deleting it could lose a required rebind
    // after another handle has reused the physical scaling buffer.
    if (auto established = dyn_cast_or_null<pto::TGetScaleAddrOp>(matchingCopy->getPrevNode())) {
        const bool matches = established.getSrc() == op.getSrc() && established.getDst() == op.getDst();
        if (matches) {
            return success();
        }
    }
    const StringRef copyName = isa<pto::TExtractOp>(matchingCopy) ? "TEXTRACT" : "TMOV";
    // The binding changes the destination address, including for uses captured
    // by nested regions. Neither crossing those uses nor silently keeping the
    // binding after the copy establishes safe backend semantics.
    if (Operation* use = findInterveningUseOfDst(matchingCopy, op, op.getDst())) {
        auto diagnostic = op.emitOpError("cannot safely establish scaling address before matching ");
        diagnostic << copyName << ": intervening destination use";
        diagnostic.attachNote(use->getLoc()) << "intervening destination use: " << use->getName();
        diagnostic.attachNote(matchingCopy->getLoc()) << "matching " << copyName << " is here";
        return failure();
    }
    llvm::SmallPtrSet<Operation*, kRiskyOpReserveSize> collected;
    SmallVector<Operation*, kRiskyOpReserveSize> dependencies;
    Operation* blockingDef = nullptr;
    for (Value operand : op->getOperands()) {
        const auto result = collectScaleAddrDependencies(
            operand, matchingCopy, op, dominance, collected, dependencies, blockingDef);
        if (failed(result)) {
            auto diagnostic = op.emitOpError("cannot safely establish scaling address before matching ");
            diagnostic << copyName << ": operand dependency cannot be hoisted";
            if (blockingDef) {
                diagnostic.attachNote(blockingDef->getLoc()) << "blocking dependency: " << blockingDef->getName();
            }
            diagnostic.attachNote(matchingCopy->getLoc()) << "matching " << copyName << " is here";
            return failure();
        }
    }
    for (Operation* dependency : dependencies) {
        dependency->moveBefore(matchingCopy);
    }
    op->moveBefore(matchingCopy);
    return success();
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
  LogicalResult hoistScaleAddrOpsBeforeCopy(func::FuncOp func) const;
  LogicalResult normalizeRiskyVecVecTMov(func::FuncOp func) const;
  LogicalResult rewriteRiskyVecVecTMov(IRRewriter &rewriter, pto::TMovOp op) const;
  void verifyNoResidualRiskyTMov(func::FuncOp func);
};

LogicalResult PTOA5NormalizeTMovPass::hoistScaleAddrOpsBeforeCopy(func::FuncOp func) const
{
    SmallVector<pto::TGetScaleAddrOp, kRiskyOpReserveSize> scaleAddrOps;
    DominanceInfo dominance(func);
    func.walk([&](pto::TGetScaleAddrOp op) { scaleAddrOps.push_back(op); });
    for (pto::TGetScaleAddrOp op : scaleAddrOps) {
        auto matchingCopy = findMatchingScaleTileCopy(op);
        if (!matchingCopy) {
            continue;
        }
        const bool involvesExtract = isa<pto::TExtractOp>(matchingCopy) || mayBindScaleExtract(op);
        const bool preserveBinding = involvesExtract ? !canHoistScaleExtractBinding(op, matchingCopy)
                                                     : hasEstablishedScaleAddrBinding(op, matchingCopy);
        if (preserveBinding) {
            continue;
        }
        if (failed(hoistScaleAddr(op, matchingCopy, dominance))) {
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

void PTOA5NormalizeTMovPass::runOnOperation()
{
    func::FuncOp func = getOperation();
    if (!isTargetArchA5(func.getOperation())) {
        return;
    }

    if (failed(hoistScaleAddrOpsBeforeCopy(func))) {
        signalPassFailure();
        return;
    }

    if (failed(normalizeRiskyVecVecTMov(func))) {
        signalPassFailure();
        return;
    }

    verifyNoResidualRiskyTMov(func);
}

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOA5NormalizeTMovPass() {
  return std::make_unique<PTOA5NormalizeTMovPass>();
}

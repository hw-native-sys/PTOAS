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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

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
  if (pto::classifyTMovForm(op.getFp()) == pto::TMovForm::XToZz)
    return false;
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

static bool hasInterveningUsesOfDst(Operation *start, Operation *end,
                                    Value dst) {
  for (Operation *cursor = start->getNextNode(); cursor && cursor != end;
       cursor = cursor->getNextNode()) {
    for (Value operand : cursor->getOperands()) {
      if (operand == dst) {
        return true;
      }
    }
  }
  return false;
}

static pto::TMovOp findMatchingScaleTileTMov(pto::TGetScaleAddrOp op) {
  Value dst = op.getDst();
  for (Operation *cursor = op->getPrevNode(); cursor; cursor = cursor->getPrevNode()) {
    auto mov = dyn_cast<pto::TMovOp>(cursor);
    if (!mov || mov.getDst() != dst || !isA5ScaleTileTMov(mov)) {
      continue;
    }
    if (hasInterveningUsesOfDst(mov, op, dst)) {
      return {};
    }
    return mov;
  }
  return {};
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
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!isTargetArchA5(func.getOperation())) {
      return;
    }

    SmallVector<pto::TGetScaleAddrOp, kRiskyOpReserveSize> scaleAddrOps;
    func.walk([&](pto::TGetScaleAddrOp op) { scaleAddrOps.push_back(op); });
    for (pto::TGetScaleAddrOp op : scaleAddrOps) {
      auto matchingTMov = findMatchingScaleTileTMov(op);
      if (!matchingTMov) {
        continue;
      }
      op->moveBefore(matchingTMov);
    }

    SmallVector<pto::TMovOp, kRiskyOpReserveSize> riskyOps;
    func.walk([&](pto::TMovOp op) {
      if (isA5RiskyVecVecColMajorTMov(op)) {
        riskyOps.push_back(op);
      }
    });

    IRRewriter rewriter(func.getContext());
    for (pto::TMovOp op : riskyOps) {
      auto srcTb = cast<pto::TileBufType>(op.getSrc().getType());
      auto dstTb = cast<pto::TileBufType>(op.getDst().getType());

      FailureOr<pto::TileBufType> srcRowTy =
          buildRowMajorReinterpretType(func.getContext(), srcTb);
      FailureOr<pto::TileBufType> dstRowTy =
          buildRowMajorReinterpretType(func.getContext(), dstTb);
      if (failed(srcRowTy) || failed(dstRowTy)) {
        op.emitOpError(
            "cannot normalize A5 vec->vec col_major TMOV: requires static 2D "
            "tile_buf shape/valid_shape for treshape reinterpret");
        signalPassFailure();
        return;
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
        signalPassFailure();
        return;
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
    }

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
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOA5NormalizeTMovPass() {
  return std::make_unique<PTOA5NormalizeTMovPass>();
}

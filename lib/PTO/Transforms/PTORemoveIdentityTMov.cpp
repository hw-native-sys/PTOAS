// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTORemoveIdentityTMov.cpp -----------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOREMOVEIDENTITYTMOV
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static std::optional<unsigned> getIntegerLikeBitWidth(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    return intTy.getWidth();
  }
  if (isa<IndexType>(type)) {
    return mlir::pto::kValue64;
  }
  return std::nullopt;
}

static std::optional<APInt> tryEvalIntegerLikeConstant(Value value);

static std::optional<APInt> evalSignedCast(Value input, Type resultType) {
  std::optional<APInt> inputValue = tryEvalIntegerLikeConstant(input);
  std::optional<unsigned> resultWidth = getIntegerLikeBitWidth(resultType);
  if (!inputValue || !resultWidth) {
    return std::nullopt;
  }
  return inputValue->sextOrTrunc(*resultWidth);
}

static std::optional<APInt> evalUnsignedCast(Value input, Type resultType) {
  std::optional<APInt> inputValue = tryEvalIntegerLikeConstant(input);
  std::optional<unsigned> resultWidth = getIntegerLikeBitWidth(resultType);
  if (!inputValue || !resultWidth) {
    return std::nullopt;
  }
  return inputValue->zextOrTrunc(*resultWidth);
}

static std::optional<APInt> evalTruncCast(Value input, Type resultType) {
  std::optional<APInt> inputValue = tryEvalIntegerLikeConstant(input);
  std::optional<unsigned> resultWidth = getIntegerLikeBitWidth(resultType);
  if (!inputValue || !resultWidth || *resultWidth > inputValue->getBitWidth()) {
    return std::nullopt;
  }
  return inputValue->trunc(*resultWidth);
}

static std::optional<APInt> tryEvalIntegerLikeConstant(Value value) {
  if (!value) {
    return std::nullopt;
  }

  APInt apInt;
  if (matchPattern(value, m_ConstantInt(&apInt))) {
    std::optional<unsigned> width = getIntegerLikeBitWidth(value.getType());
    if (!width) {
      return std::nullopt;
    }
    return apInt.sextOrTrunc(*width);
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }

  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp)) {
    return evalSignedCast(castOp.getIn(), castOp.getType());
  }
  if (auto castOp = dyn_cast<arith::IndexCastUIOp>(defOp)) {
    return evalUnsignedCast(castOp.getIn(), castOp.getType());
  }
  if (auto castOp = dyn_cast<arith::ExtSIOp>(defOp)) {
    return evalSignedCast(castOp.getIn(), castOp.getType());
  }
  if (auto castOp = dyn_cast<arith::ExtUIOp>(defOp)) {
    return evalUnsignedCast(castOp.getIn(), castOp.getType());
  }
  if (auto castOp = dyn_cast<arith::TruncIOp>(defOp)) {
    return evalTruncCast(castOp.getIn(), castOp.getType());
  }

  return std::nullopt;
}

static std::optional<int64_t> tryEvalI64Constant(Value value) {
  std::optional<APInt> apInt = tryEvalIntegerLikeConstant(value);
  if (!apInt || apInt->getBitWidth() > mlir::pto::kValue64) {
    return std::nullopt;
  }
  return apInt->getSExtValue();
}

static bool hasOnlyCurrentOpUses(Value value, Operation *currentOp) {
  return llvm::all_of(value.getUses(), [currentOp](OpOperand &use) {
    return use.getOwner() == currentOp;
  });
}

static bool hasNoResultUses(Operation *op) {
  return llvm::all_of(op->getResults(),
                      [](OpResult result) { return result.use_empty(); });
}

static bool isDeadDstTMov(TMovOp op) {
  return hasOnlyCurrentOpUses(op.getDst(), op) && hasNoResultUses(op);
}

static const BaseMemInfo *
getSingleMemInfo(const Buffer2MemInfoMap &buffer2MemInfoMap, Value value) {
  auto it = buffer2MemInfoMap.find(value);
  if (it == buffer2MemInfoMap.end() || it->second.size() != 1) {
    return nullptr;
  }
  return it->second.front().get();
}

static std::optional<int64_t>
tryGetConcreteRootAddress(const BaseMemInfo *info) {
  if (!info) {
    return std::nullopt;
  }

  if (auto direct = tryEvalI64Constant(info->rootBuffer)) {
    return direct;
  }

  Operation *defOp = info->rootBuffer.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }

  if (auto alloc = dyn_cast<pto::AllocTileOp>(defOp)) {
    return tryEvalI64Constant(alloc.getAddr());
  }

  return std::nullopt;
}

static bool hasDynamicStaticList(ArrayRef<int64_t> values) {
  return llvm::any_of(
      values, [](int64_t value) { return value == ShapedType::kDynamic; });
}

static bool isStaticallyAddressableValue(Value value) {
  int depth = 0;
  constexpr int kMaxDepth = 32;
  while (value && depth++ < kMaxDepth) {
    Operation *defOp = value.getDefiningOp();
    if (!defOp) {
      return false;
    }

    if (auto subView = dyn_cast<memref::SubViewOp>(defOp)) {
      if (hasDynamicStaticList(subView.getStaticOffsets()) ||
          hasDynamicStaticList(subView.getStaticSizes()) ||
          hasDynamicStaticList(subView.getStaticStrides())) {
        return false;
      }
      value = subView.getSource();
      continue;
    }

    if (isa<memref::ReinterpretCastOp>(defOp)) {
      return false;
    }

    if (auto cast = dyn_cast<memref::CastOp>(defOp)) {
      value = cast.getSource();
      continue;
    }
    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(defOp)) {
      value = collapse.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<memref::ExpandShapeOp>(defOp)) {
      value = expand.getSrc();
      continue;
    }
    if (auto view = dyn_cast<memref::ViewOp>(defOp)) {
      if (view.getByteShift()) {
        return false;
      }
      value = view.getSource();
      continue;
    }

    return true;
  }

  return false;
}

static bool hasExactSameAddressRange(const BaseMemInfo *srcInfo,
                                     const BaseMemInfo *dstInfo) {
  if (!srcInfo || !dstInfo) {
    return false;
  }
  if (srcInfo->scope != dstInfo->scope) {
    return false;
  }
  if (srcInfo->allocateSize == 0 || dstInfo->allocateSize == 0) {
    return false;
  }
  if (srcInfo->allocateSize != dstInfo->allocateSize) {
    return false;
  }
  if (srcInfo->baseAddresses.empty() || dstInfo->baseAddresses.empty()) {
    return false;
  }
  if (srcInfo->baseAddresses != dstInfo->baseAddresses) {
    return false;
  }
  return true;
}

static bool hasSameConcreteAddressRange(const BaseMemInfo *srcInfo,
                                        const BaseMemInfo *dstInfo) {
  if (!hasExactSameAddressRange(srcInfo, dstInfo)) {
    return false;
  }
  if (srcInfo->rootBuffer == dstInfo->rootBuffer) {
    return true;
  }
  auto srcRootAddr = tryGetConcreteRootAddress(srcInfo);
  auto dstRootAddr = tryGetConcreteRootAddress(dstInfo);
  return srcRootAddr && dstRootAddr && *srcRootAddr == *dstRootAddr;
}

static Operation *getAncestorInBlock(Operation *op, Block *block) {
  for (Operation *cur = op; cur; cur = cur->getParentOp()) {
    if (cur->getBlock() == block) {
      return cur;
    }
  }
  return nullptr;
}

static bool hasUseAfterOp(Value value, Operation *currentOp) {
  Block *block = currentOp->getBlock();
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == currentOp) {
      continue;
    }
    Operation *ancestor = getAncestorInBlock(owner, block);
    if (!ancestor) {
      return true;
    }
    if (ancestor != currentOp && currentOp->isBeforeInBlock(ancestor)) {
      return true;
    }
  }
  return false;
}

static bool hasLaterUseOfSameAddressRange(
    TMovOp op, const BaseMemInfo *dstInfo,
    const Buffer2MemInfoMap &buffer2MemInfoMap) {
  for (const auto &entry : buffer2MemInfoMap) {
    // Later reads of the source itself do not make this no-op TMOV a bridge.
    if (entry.first == op.getSrc()) {
      continue;
    }
    bool sameRange = llvm::any_of(entry.second, [&](const auto &info) {
      return hasSameConcreteAddressRange(info.get(), dstInfo);
    });
    if (sameRange && hasUseAfterOp(entry.first, op)) {
      return true;
    }
  }
  return false;
}

static bool hasPlainTMovSemantics(TMovOp op) {
  return !op.getFp() && !op.getPreQuantScalar() && !op.getAccToVecModeAttr() &&
         op.getReluPreMode() == ReluPreMode::NoRelu;
}

static bool hasCompatibleIdentityTypes(TMovOp op) {
  if (op.getSrc().getType() != op.getDst().getType()) {
    return false;
  }
  for (OpResult result : op->getResults()) {
    if (result.getType() != op.getDst().getType()) {
      return false;
    }
  }
  return true;
}

static bool hasLowPrecisionElement(Value value) {
  if (auto tileTy = dyn_cast<TileBufType>(value.getType())) {
    return isPTOLowPrecisionType(tileTy.getElementType());
  }
  if (auto memrefTy = dyn_cast<MemRefType>(value.getType())) {
    return isPTOLowPrecisionType(memrefTy.getElementType());
  }
  return false;
}

static bool touchesLowPrecisionElement(TMovOp op) {
  if (hasLowPrecisionElement(op.getSrc()) || hasLowPrecisionElement(op.getDst())) {
    return true;
  }
  return llvm::any_of(op->getResults(), [](OpResult result) {
    if (auto tileTy = dyn_cast<TileBufType>(result.getType())) {
      return isPTOLowPrecisionType(tileTy.getElementType());
    }
    if (auto memrefTy = dyn_cast<MemRefType>(result.getType())) {
      return isPTOLowPrecisionType(memrefTy.getElementType());
    }
    return false;
  });
}

static bool
isIdentityTMovByMemInfo(TMovOp op, const Buffer2MemInfoMap &buffer2MemInfoMap) {
  Value src = op.getSrc();
  Value dst = op.getDst();
  if (!isStaticallyAddressableValue(src) || !isStaticallyAddressableValue(dst)) {
    return false;
  }

  const BaseMemInfo *srcInfo = getSingleMemInfo(buffer2MemInfoMap, src);
  const BaseMemInfo *dstInfo = getSingleMemInfo(buffer2MemInfoMap, dst);
  if (!hasSameConcreteAddressRange(srcInfo, dstInfo)) {
    return false;
  }

  return isDeadDstTMov(op) &&
         !hasLaterUseOfSameAddressRange(op, dstInfo, buffer2MemInfoMap);
}

struct PTORemoveIdentityTMovPass
    : public mlir::pto::impl::PTORemoveIdentityTMovBase<
          PTORemoveIdentityTMovPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<TMovOp, mlir::pto::kValue16> identityMoves;
    SmallVector<TMovOp, mlir::pto::kValue16> memInfoCandidates;

    func.walk([&](TMovOp op) {
      if (!hasPlainTMovSemantics(op) || !hasCompatibleIdentityTypes(op) ||
          touchesLowPrecisionElement(op)) {
        return;
      }
      if (op.getSrc() == op.getDst()) {
        identityMoves.push_back(op);
        return;
      }
      memInfoCandidates.push_back(op);
    });

    if (!memInfoCandidates.empty()) {
      MemoryDependentAnalyzer memAnalyzer;
      SyncIRs syncIR;
      Buffer2MemInfoMap buffer2MemInfoMap;
      PTOIRTranslator translator(syncIR, memAnalyzer, buffer2MemInfoMap, func,
                                 SyncAnalysisMode::NORMALSYNC);
      translator.Build();

      for (TMovOp op : memInfoCandidates) {
        if (isIdentityTMovByMemInfo(op, buffer2MemInfoMap)) {
          identityMoves.push_back(op);
        }
      }
    }

    for (TMovOp op : identityMoves) {
      for (OpResult result : op->getResults()) {
        result.replaceAllUsesWith(op.getDst());
      }
      op.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTORemoveIdentityTMovPass() {
  return std::make_unique<PTORemoveIdentityTMovPass>();
}

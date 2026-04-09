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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOCANONICALIZESUBVIEWFORTLOAD
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static constexpr llvm::StringLiteral kLayoutAttrName = "layout";
static constexpr llvm::StringLiteral kSingletonAxisPermutationAttrName =
    "pto.singleton_axis_permutation";

static Value peelUnrealized(Value v) {
  if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
    return castOp.getOperand(0);
  return v;
}

static std::optional<int64_t> extractStaticInt(OpFoldResult ofr) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    if (auto ia = dyn_cast<IntegerAttr>(attr))
      return ia.getInt();
    return std::nullopt;
  }
  Value v = ofr.get<Value>();
  if (auto cIdx = v.getDefiningOp<arith::ConstantIndexOp>())
    return cIdx.value();
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>())
    return cInt.value();
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  }
  return std::nullopt;
}

static std::optional<mlir::pto::Layout> getLayoutAttrFromOp(Operation *op) {
  if (!op)
    return std::nullopt;
  if (auto attr = op->getAttrOfType<mlir::pto::LayoutAttr>(kLayoutAttrName))
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

static std::optional<pto::TileBufConfigAttr>
resolveTileConfigFromValueChain(Value v) {
  v = peelUnrealized(v);
  while (Operation *def = v.getDefiningOp()) {
    if (auto bind = dyn_cast<pto::BindTileOp>(def))
      return bind.getConfigAttr();
    if (auto cast = dyn_cast<pto::PointerCastOp>(def)) {
      if (auto cfg = cast.getConfig())
        return *cfg;
      return std::nullopt;
    }
    if (auto mcast = dyn_cast<memref::CastOp>(def)) {
      v = peelUnrealized(mcast.getSource());
      continue;
    }
    if (auto rc = dyn_cast<memref::ReinterpretCastOp>(def)) {
      v = peelUnrealized(rc.getSource());
      continue;
    }
    if (auto sv = dyn_cast<memref::SubViewOp>(def)) {
      v = peelUnrealized(sv.getSource());
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

static bool isNZLikeTileConfig(pto::TileBufConfigAttr configAttr) {
  int32_t blVal = 0;
  if (auto bl = dyn_cast<BLayoutAttr>(configAttr.getBLayout()))
    blVal = static_cast<int32_t>(bl.getValue());

  int32_t slVal = 0;
  if (auto sl = dyn_cast<SLayoutAttr>(configAttr.getSLayout()))
    slVal = static_cast<int32_t>(sl.getValue());

  int32_t fractal = 0;
  if (auto fr = dyn_cast<IntegerAttr>(configAttr.getSFractalSize()))
    fractal = fr.getInt();

  return blVal == static_cast<int32_t>(BLayout::ColMajor) &&
         slVal == static_cast<int32_t>(SLayout::RowMajor) && fractal == 512;
}

static bool tracesBackThroughViewCasts(Value v, Value target) {
  Value cur = peelUnrealized(v);
  for (int guard = 0; guard < 64; ++guard) {
    if (cur == target)
      return true;
    Operation *def = cur.getDefiningOp();
    if (!def)
      return false;
    if (auto mcast = dyn_cast<memref::CastOp>(def)) {
      cur = peelUnrealized(mcast.getSource());
      continue;
    }
    if (auto rc = dyn_cast<memref::ReinterpretCastOp>(def)) {
      cur = peelUnrealized(rc.getSource());
      continue;
    }
    if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (unrealized->getNumOperands() == 0)
        return false;
      cur = peelUnrealized(unrealized.getOperand(0));
      continue;
    }
    return false;
  }
  return false;
}

static void collectUsersThroughViewCasts(Value v,
                                         SmallVectorImpl<Operation *> &out) {
  SmallVector<Value, 8> worklist;
  llvm::DenseSet<Value> visitedValues;
  llvm::DenseSet<Operation *> visitedUsers;
  worklist.push_back(v);

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!visitedValues.insert(cur).second)
      continue;
    for (Operation *u : cur.getUsers()) {
      if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(u)) {
        for (Value r : unrealized->getResults())
          worklist.push_back(r);
        continue;
      }
      if (auto mcast = dyn_cast<memref::CastOp>(u)) {
        worklist.push_back(mcast.getResult());
        continue;
      }
      if (auto rc = dyn_cast<memref::ReinterpretCastOp>(u)) {
        worklist.push_back(rc.getResult());
        continue;
      }
      if (visitedUsers.insert(u).second)
        out.push_back(u);
    }
  }
}

static bool isNdDnToNzLikeTLoad(pto::TLoadOp load) {
  if (!load.getDst())
    return false;

  auto gtLayout =
      resolveLayoutForGlobalTensor(load.getOperation(), load.getSrc());
  if (!gtLayout ||
      (*gtLayout != mlir::pto::Layout::ND &&
       *gtLayout != mlir::pto::Layout::DN))
    return false;

  auto tileCfg = resolveTileConfigFromValueChain(load.getDst());
  if (!tileCfg)
    return false;
  return isNZLikeTileConfig(*tileCfg);
}

static bool shouldCanonicalizeSubviewForNdDnToNz(memref::SubViewOp sv) {
  SmallVector<Operation *, 8> users;
  collectUsersThroughViewCasts(sv.getResult(), users);
  bool sawTarget = false;

  for (Operation *user : users) {
    auto load = dyn_cast<pto::TLoadOp>(user);
    if (!load)
      return false;
    if (!tracesBackThroughViewCasts(load.getSrc(), sv.getResult()))
      continue;
    if (!isNdDnToNzLikeTLoad(load))
      return false;
    sawTarget = true;
  }
  return sawTarget;
}

static std::optional<SmallVector<int64_t, 8>>
computeSingletonFirstPermutation(memref::SubViewOp sv) {
  auto resTy = dyn_cast<MemRefType>(sv.getResult().getType());
  if (!resTy)
    return std::nullopt;

  const int rank = resTy.getRank();
  if (rank <= 2)
    return std::nullopt;

  auto mixedSizes = sv.getMixedSizes();
  auto resShape = resTy.getShape();
  SmallVector<bool, 8> staticSingletonDims;
  staticSingletonDims.reserve(rank);

  int nonSingletonCount = 0;
  for (int i = 0; i < rank; ++i) {
    std::optional<int64_t> staticDim;
    if (i < static_cast<int>(mixedSizes.size()))
      staticDim = extractStaticInt(mixedSizes[i]);
    if (!staticDim && resShape[i] != ShapedType::kDynamic)
      staticDim = resShape[i];

    bool isSingleton = staticDim && *staticDim == 1;
    staticSingletonDims.push_back(isSingleton);
    if (!isSingleton)
      ++nonSingletonCount;
  }

  if (nonSingletonCount > 2)
    return std::nullopt;

  SmallVector<int64_t, 8> permutation;
  permutation.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    if (staticSingletonDims[i])
      permutation.push_back(i);
  }
  for (int i = 0; i < rank; ++i) {
    if (!staticSingletonDims[i])
      permutation.push_back(i);
  }

  bool changed = false;
  for (int i = 0; i < rank; ++i) {
    if (permutation[i] != i) {
      changed = true;
      break;
    }
  }
  if (!changed)
    return std::nullopt;

  return permutation;
}

struct PTOCanonicalizeSubviewForTLoadPass
    : public mlir::pto::impl::PTOCanonicalizeSubviewForTLoadBase<
          PTOCanonicalizeSubviewForTLoadPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    func.walk([&](memref::SubViewOp sv) {
      sv->removeAttr(kSingletonAxisPermutationAttrName);

      auto perm = computeSingletonFirstPermutation(sv);
      if (!perm)
        return;
      if (!shouldCanonicalizeSubviewForNdDnToNz(sv))
        return;

      sv->setAttr(kSingletonAxisPermutationAttrName,
                  DenseI64ArrayAttr::get(ctx, *perm));
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOCanonicalizeSubviewForTLoadPass() {
  return std::make_unique<PTOCanonicalizeSubviewForTLoadPass>();
}

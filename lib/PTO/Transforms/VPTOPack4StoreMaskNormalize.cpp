// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOPack4StoreMaskNormalize.cpp - Byte granular packed store masks --===//
//
// Post-VMI-to-VPTO pass that rewrites the predicate of a `pto.vsts` operation
// storing packed 4-bit values with the `PK4_B32` distribution from the 32-bit
// granular spelling used by the lowering into the byte granular spelling the
// A5 model executes at full rate.
//
// The rewrite is purely a spelling change of one predicate:
//
//   %m32 = pto.pset_b32 "PAT_VL32" : !pto.mask<b32>
//   pto.vsts %v, %dst[%off], %m32 {dist = "PK4_B32"} : ...
//
// becomes
//
//   %m8  = pto.pset_b8 "PAT_VL128" : !pto.mask<b8>
//   %m32 = pto.pbitcast %m8 : !pto.mask<b8> -> !pto.mask<b32>
//   pto.vsts %v, %dst[%off], %m32 {dist = "PK4_B32"} : ...
//
// One b32 lane covers the four bytes of one 32-bit packed store group, so four
// consecutive b8 lanes cover exactly the same bytes. The store operand type and
// the store semantics are therefore unchanged; only the predicate producer
// changes. Keeping the store operand a b32 mask is required by the pto.vsts
// verifier, which fixes the mask granularity of a PK4_B32 store to b32.
//
// This pass deliberately runs after the lowering instead of inside it: the
// byte granular predicate is a model and dialect compatibility concern, not a
// property of the general VMI-to-VPTO mapping.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOPACK4STOREMASKNORMALIZE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

/// Distribution of a store that packs four 4-bit values into one 32-bit group.
constexpr StringLiteral kPacked4BitStoreDist("PK4_B32");

/// Spell the byte granular pattern that covers the same bytes as the 32-bit
/// granular pattern b32Pattern. An empty result means the pattern has no byte
/// granular spelling and the caller must keep the original predicate.
StringRef getB8SpellingOfB32Prefix(StringRef b32Pattern) {
  return llvm::StringSwitch<StringRef>(b32Pattern)
      .Case("PAT_VL64", "PAT_ALL")
      .Case("PAT_ALL", "PAT_ALL")
      .Case("PAT_VL32", "PAT_VL128")
      .Case("PAT_VL16", "PAT_VL64")
      .Case("PAT_VL8", "PAT_VL32")
      .Case("PAT_VL4", "PAT_VL16")
      .Case("PAT_VL2", "PAT_VL8")
      .Case("PAT_VL1", "PAT_VL4")
      .Default("");
}

/// Return the value element type of a packed 4-bit store, or a null type when
/// the store is not a PK4_B32 store of packed 4-bit values.
Type getPacked4BitStoreElementType(VstsOp store) {
  std::optional<StringRef> dist = store.getDist();
  if (!dist || *dist != kPacked4BitStoreDist) {
    return Type();
  }
  auto valueType = dyn_cast<VRegType>(store.getValue().getType());
  if (!valueType) {
    return Type();
  }
  Type elementType = valueType.getElementType();
  return pto::isPTOFloat4PackedType(elementType) ? elementType : Type();
}

struct NormalizePacked4BitStoreMaskPattern : OpRewritePattern<VstsOp> {
  using OpRewritePattern<VstsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(VstsOp store,
                                PatternRewriter &rewriter) const override {
    if (!getPacked4BitStoreElementType(store)) {
      return failure();
    }
    // Only a directly produced static pset_b32 prefix has a byte granular
    // spelling. Runtime pto.plt_* predicates, pge masks, and patterns without a
    // legal byte spelling stay 32-bit granular.
    auto b32Mask = store.getMask().getDefiningOp<PsetB32Op>();
    if (!b32Mask) {
      return failure();
    }
    StringRef b8Pattern = getB8SpellingOfB32Prefix(b32Mask.getPattern());
    if (b8Pattern.empty()) {
      return failure();
    }

    MLIRContext *ctx = rewriter.getContext();
    Location loc = store.getLoc();
    // A pset_b32 that feeds only this store can be replaced in place, so the
    // rewritten chain inherits its position and its other uses stay untouched.
    // A shared pset_b32 keeps its 32-bit granular spelling for the remaining
    // users, and the store gets a fresh byte granular mask of its own.
    const bool b32MaskFeedsOnlyThisStore = b32Mask->hasOneUse();
    rewriter.setInsertionPoint(b32MaskFeedsOnlyThisStore
                                   ? b32Mask.getOperation()
                                   : store.getOperation());

    Value b8Mask = rewriter
                       .create<PsetB8Op>(loc, MaskType::get(ctx, "b8"),
                                         rewriter.getStringAttr(b8Pattern))
                       .getResult();
    Value b32View = rewriter
                        .create<PbitcastOp>(loc, MaskType::get(ctx, "b32"),
                                            b8Mask)
                        .getResult();
    if (b32MaskFeedsOnlyThisStore) {
      rewriter.replaceOp(b32Mask, b32View);
      return success();
    }
    rewriter.modifyOpInPlace(
        store, [&] { store.getMaskMutable().assign(b32View); });
    return success();
  }
};

struct VPTOPack4StoreMaskNormalizePass
    : public pto::impl::VPTOPack4StoreMaskNormalizeBase<
          VPTOPack4StoreMaskNormalizePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<NormalizePacked4BitStoreMaskPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOPack4StoreMaskNormalizePass() {
  return std::make_unique<VPTOPack4StoreMaskNormalizePass>();
}

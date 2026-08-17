// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTORematerializeFixpipeVectorQuant.cpp -----------------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOREMATERIALIZEFIXPIPEVECTORQUANT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

struct PTORematerializeFixpipeVectorQuantPass
    : public mlir::pto::impl::PTORematerializeFixpipeVectorQuantBase<
          PTORematerializeFixpipeVectorQuantPass> {
  using Base =
      mlir::pto::impl::PTORematerializeFixpipeVectorQuantBase<
          PTORematerializeFixpipeVectorQuantPass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<Operation *> eraseList;

    auto processBlock = [&](auto &&self, Block &block) -> LogicalResult {
      llvm::DenseMap<int32_t, SetQuantVectorOp> activeVectorById;
      SmallVector<Operation *> originalOps;
      for (Operation &op : block) {
        originalOps.push_back(&op);
      }

      for (Operation *op : originalOps) {
        if (auto setQuantVector = dyn_cast<SetQuantVectorOp>(op)) {
          activeVectorById[setQuantVector.getId()] = setQuantVector;
          eraseList.push_back(op);
        } else if (auto tpush = dyn_cast<TPushOp>(op)) {
          auto accPushEpilogue =
              getPipeInitAccPushEpilogue(getPipeInitDef(tpush.getPipeHandle()));
          auto pipeId = getFrontendPipeIdFromHandle(tpush.getPipeHandle());
          if (accPushEpilogue && pipeId &&
              isVectorFixpipeQuant(accPushEpilogue.getQuant())) {
            auto it = activeVectorById.find(*pipeId);
            if (it != activeVectorById.end()) {
              OpBuilder builder(tpush);
              builder.clone(*it->second.getOperation());
            }
          }
        }

        for (Region &region : op->getRegions()) {
          for (Block &nestedBlock : region) {
            if (failed(self(self, nestedBlock))) {
              return failure();
            }
          }
        }
      }
      return success();
    };

    for (Block &block : funcOp.getBlocks()) {
      if (failed(processBlock(processBlock, block))) {
        signalPassFailure();
        return;
      }
    }

    for (Operation *op : eraseList) {
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTORematerializeFixpipeVectorQuantPass() {
  return std::make_unique<PTORematerializeFixpipeVectorQuantPass>();
}

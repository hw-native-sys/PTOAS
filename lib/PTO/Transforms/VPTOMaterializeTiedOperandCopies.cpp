// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOMaterializeTiedOperandCopies.cpp - Materialize VPTO copies ----===//
//
// This pass makes physical copies explicit before destructive vector
// operations whose result must reuse one input register. It is deliberately
// independent of the VPTO scheduler.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/VPTOPhysicalRegister.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOMATERIALIZETIEDOPERANDCOPIES
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

struct TiedUse {
  Operation *operation = nullptr;
  unsigned operandIndex = 0;
};

using TiedUseGroups = llvm::MapVector<Value, SmallVector<TiedUse, 4>>;

static StringAttr findTargetArchitecture(Operation *operation) {
  for (Operation *current = operation; current;
       current = current->getParentOp()) {
    auto module = dyn_cast<ModuleOp>(current);
    if (!module) {
      continue;
    }
    if (auto target =
            module->getAttrOfType<StringAttr>(pto::kPTOTargetArchAttrName)) {
      return target;
    }
  }
  return {};
}

static bool isInsideVectorScope(Operation *operation) {
  return operation->getParentOfType<VecScopeOp>() ||
         operation->getParentOfType<StrictVecScopeOp>();
}

static FailureOr<TiedUse> validateTiedUse(Operation *operation) {
  auto tied = dyn_cast<VPTOTiedOperandOpInterface>(operation);
  if (!tied) {
    return failure();
  }

  unsigned operandIndex = tied.getTiedOperandIndex();
  unsigned resultIndex = tied.getTiedResultIndex();
  if (operandIndex >= operation->getNumOperands()) {
    operation->emitOpError("tied-operand interface returned operand index ")
        << operandIndex << " for " << operation->getNumOperands()
        << " operands";
    return failure();
  }
  if (resultIndex >= operation->getNumResults()) {
    operation->emitOpError("tied-operand interface returned result index ")
        << resultIndex << " for " << operation->getNumResults() << " results";
    return failure();
  }

  Type operandType = operation->getOperand(operandIndex).getType();
  Type resultType = operation->getResult(resultIndex).getType();
  bool hasNonVectorType =
      !isa<VRegType>(operandType) || !isa<VRegType>(resultType);
  if (hasNonVectorType) {
    operation->emitOpError(
        "tied operand and result must both be single physical !pto.vreg values");
    return failure();
  }
  if (operandType != resultType) {
    operation->emitOpError(
        "tied operand and result must have identical !pto.vreg types");
    return failure();
  }
  return TiedUse{operation, operandIndex};
}

static LogicalResult collectTiedUseGroups(func::FuncOp function,
                                          TiedUseGroups &groups) {
  WalkResult result = function.walk([&](Operation *operation) {
    bool isTiedOperation = isa<VPTOTiedOperandOpInterface>(operation);
    bool isInVectorScope = isInsideVectorScope(operation);
    if (!isInVectorScope || !isTiedOperation) {
      return WalkResult::advance();
    }
    FailureOr<TiedUse> tiedUse = validateTiedUse(operation);
    if (failed(tiedUse)) {
      return WalkResult::interrupt();
    }
    Value tiedOperand = operation->getOperand(tiedUse->operandIndex);
    groups[getPhysicalRegisterViewRoot(tiedOperand)].push_back(*tiedUse);
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

struct RootUseAnalysis {
  bool allUsesInBlock = true;
  SmallVector<Operation *, 8> materialUsers;
};

static RootUseAnalysis analyzeRootUses(Value root, Block *expectedBlock) {
  RootUseAnalysis analysis;
  SmallVector<Value, 8> worklist{root};
  llvm::DenseSet<Value> visitedValues;
  llvm::SmallPtrSet<Operation *, 8> visitedMaterialUsers;

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visitedValues.insert(value).second) {
      continue;
    }
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      bool isViewOperand =
          isPhysicalRegisterView(owner) && use.getOperandNumber() == 0;
      if (isViewOperand) {
        analysis.allUsesInBlock &= owner->getBlock() == expectedBlock;
        worklist.append(owner->getResults().begin(), owner->getResults().end());
        continue;
      }

      analysis.allUsesInBlock &= owner->getBlock() == expectedBlock;
      if (visitedMaterialUsers.insert(owner).second) {
        analysis.materialUsers.push_back(owner);
      }
    }
  }
  return analysis;
}

static Operation *findLastMaterialUser(const RootUseAnalysis &analysis,
                                       Block *block) {
  if (!analysis.allUsesInBlock) {
    return nullptr;
  }
  Operation *last = nullptr;
  for (Operation *user : analysis.materialUsers) {
    Block *userBlock = user->getBlock();
    if (userBlock != block) {
      return nullptr;
    }
    if (!last || last->isBeforeInBlock(user)) {
      last = user;
    }
  }
  return last;
}

static std::optional<TiedUse> chooseOwner(Value root,
                                          ArrayRef<TiedUse> tiedUses) {
  if (tiedUses.empty()) {
    return std::nullopt;
  }
  Block *block = tiedUses.front().operation->getBlock();
  RootUseAnalysis analysis = analyzeRootUses(root, block);
  Operation *lastMaterialUser = findLastMaterialUser(analysis, block);
  if (!lastMaterialUser) {
    return std::nullopt;
  }
  for (const TiedUse &tiedUse : tiedUses) {
    if (tiedUse.operation == lastMaterialUser) {
      return tiedUse;
    }
  }
  return std::nullopt;
}

static bool isSameTiedUse(const TiedUse &lhs, const TiedUse &rhs) {
  return lhs.operation == rhs.operation && lhs.operandIndex == rhs.operandIndex;
}

static void materializeCopies(TiedUseGroups &groups, MLIRContext *context) {
  OpBuilder builder(context);
  for (auto &entry : groups) {
    Value root = entry.first;
    SmallVectorImpl<TiedUse> &tiedUses = entry.second;
    std::optional<TiedUse> owner = chooseOwner(root, tiedUses);
    for (const TiedUse &tiedUse : tiedUses) {
      if (owner && isSameTiedUse(*owner, tiedUse)) {
        continue;
      }
      Value operand = tiedUse.operation->getOperand(tiedUse.operandIndex);
      builder.setInsertionPoint(tiedUse.operation);
      auto copy = builder.create<VmovOp>(tiedUse.operation->getLoc(),
                                         operand.getType(), operand);
      tiedUse.operation->setOperand(tiedUse.operandIndex, copy.getResult());
    }
  }
}

struct VPTOMaterializeTiedOperandCopiesPass
    : public pto::impl::VPTOMaterializeTiedOperandCopiesBase<
          VPTOMaterializeTiedOperandCopiesPass> {
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    StringAttr target = findTargetArchitecture(function);
    if (!target) {
      function.emitError(
          "VPTO tied-copy materialization requires target architecture 'a5', "
          "but neither this function's module nor an enclosing module defines "
          "'pto.target_arch'");
      return signalPassFailure();
    }
    StringRef targetName = target.getValue();
    if (targetName != "a5") {
      function.emitError(
          "VPTO tied-copy materialization requires target architecture 'a5', "
          "but module targets '")
          << targetName << "'";
      return signalPassFailure();
    }

    TiedUseGroups groups;
    if (failed(collectTiedUseGroups(function, groups))) {
      return signalPassFailure();
    }
    materializeCopies(groups, &getContext());
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createVPTOMaterializeTiedOperandCopiesPass() {
  return std::make_unique<VPTOMaterializeTiedOperandCopiesPass>();
}

// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- InferPTOMemScope.cpp - Infer Memory Scope for pto Ops ------------===//
//===----------------------------------------------------------------------===//

#include "InferPTOMemScope.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"


#include "llvm/ADT/TypeSwitch.h"

#include <tuple>

#define DEBUG_TYPE "PTO-infer-mem-scope"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_INFERPTOMEMSCOPE
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace pto;

namespace {
static std::optional<memref::AllocOp> requireRootAlloc(Operation *op, Value value,
                                                       StringRef valueName) {
  auto alloc = tracebackMemRefToAlloc(value);
  if (!alloc.has_value()) {
    emitError(op->getLoc()) << "Cannot find root memref.alloc for " << valueName
                            << " of this op.";
  }
  return alloc;
}

static LogicalResult propagateAllocScope(Operation *op, Value value,
                                         StringRef valueName,
                                         const AddressSpaceAttr &targetScope,
                                         MemScopeInferAndPropagateHelper &helper) {
  auto alloc = requireRootAlloc(op, value, valueName);
  if (!alloc.has_value()) {
    return failure();
  }
  if (failed(helper.Run(*alloc, targetScope))) {
    return op->emitOpError()
           << "Failed to infer/propagate memory scope for " << valueName;
  }
  return success();
}

static bool hasMemRefResults(Operation *op) {
  return llvm::any_of(op->getResults(), [](OpResult result) {
    return isa<MemRefType>(result.getType());
  });
}

static LogicalResult propagateYieldScope(MemScopeInferAndPropagateHelper &helper,
                                         Value val,
                                         const AddressSpaceAttr &memrefScope,
                                         OpOperand &user) {
  auto op = cast<scf::YieldOp>(user.getOwner());
  Operation *parentOp = op->getParentOp();
  auto yieldResult = op.getOperand(user.getOperandNumber());
  auto parentResult = parentOp->getResult(user.getOperandNumber());
  auto yieldType = dyn_cast<BaseMemRefType>(yieldResult.getType());
  auto valType = dyn_cast<BaseMemRefType>(val.getType());
  if (!yieldType || !valType ||
      yieldType.getElementType() != valType.getElementType()) {
    return success();
  }
  return helper.Run(parentResult, memrefScope);
}

static LogicalResult propagateForScope(MemScopeInferAndPropagateHelper &helper,
                                       const AddressSpaceAttr &memrefScope,
                                       OpOperand &user) {
  auto op = cast<scf::ForOp>(user.getOwner());
  auto result = op.getTiedLoopResult(&user);
  auto bbArg = op.getTiedLoopRegionIterArg(&user);
  return success(helper.Run(bbArg, memrefScope).succeeded() &&
                 helper.Run(result, memrefScope).succeeded());
}

static LogicalResult propagateWhileScope(
    MemScopeInferAndPropagateHelper &helper,
    const AddressSpaceAttr &memrefScope, OpOperand &user) {
  auto op = cast<scf::WhileOp>(user.getOwner());
  unsigned index = user.getOperandNumber();
  if (index >= op.getInits().size() || index >= op.getBeforeArguments().size() ||
      index >= op.getResults().size())
    return failure();
  return success(helper.Run(op.getBeforeArguments()[index], memrefScope)
                     .succeeded() &&
                 helper.Run(op.getResults()[index], memrefScope).succeeded());
}

static LogicalResult propagateWhileConditionScope(
    MemScopeInferAndPropagateHelper &helper,
    const AddressSpaceAttr &memrefScope, OpOperand &user) {
  auto op = cast<scf::ConditionOp>(user.getOwner());
  unsigned index = user.getOperandNumber();
  if (index == 0)
    return success();
  --index;
  auto whileOp = cast<scf::WhileOp>(op->getParentOp());
  auto conditionArgs = whileOp.getConditionOp().getArgs();
  if (index >= conditionArgs.size() || index >= whileOp.getAfterArguments().size() ||
      index >= whileOp.getResults().size())
    return failure();
  return success(helper.Run(whileOp.getAfterArguments()[index], memrefScope)
                     .succeeded() &&
                 helper.Run(whileOp.getResults()[index], memrefScope).succeeded());
}

static LogicalResult
propagateViewLikeScope(MemScopeInferAndPropagateHelper &helper,
                       const AddressSpaceAttr &memrefScope, Operation *op) {
  auto result = op->getResult(0);
  return helper.Run(result, memrefScope);
}

static LogicalResult
propagateMemScopeToUser(MemScopeInferAndPropagateHelper &helper, Value val,
                        const AddressSpaceAttr &memrefScope, OpOperand &user) {
  Operation *owner = user.getOwner();
  return TypeSwitch<Operation *, LogicalResult>(owner)
      .Case<scf::YieldOp>([&](scf::YieldOp) {
        return propagateYieldScope(helper, val, memrefScope, user);
      })
      .Case<scf::ForOp>([&](scf::ForOp) {
        return propagateForScope(helper, memrefScope, user);
      })
      .Case<scf::WhileOp>([&](scf::WhileOp) {
        return propagateWhileScope(helper, memrefScope, user);
      })
      .Case<scf::ConditionOp>([&](scf::ConditionOp) {
        return propagateWhileConditionScope(helper, memrefScope, user);
      })
      .Case<memref::SubViewOp, memref::ViewOp, memref::ReinterpretCastOp,
            memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp,
            memref::ReshapeOp, memref::TransposeOp,
            memref::ExtractStridedMetadataOp>([&](auto op) {
        return propagateViewLikeScope(helper, memrefScope, op);
      })
      .Case<func::CallOp, gpu::LaunchFuncOp>([&](auto) { return success(); })
      .Default([&](Operation *op) {
        if (op->getNumResults() == 0 || !hasMemRefResults(op)) {
          return success();
        }
        op->emitOpError("Unsupported user for root alloc op.");
        return failure();
      });
}
} // namespace

LogicalResult
MemScopeInferAndPropagateHelper::propagateMemScopeToUsers(Value val) {
  auto memrefScope = getPTOAddressSpaceAttr(val.getType());
  for (OpOperand &user : val.getUses()) {
    if (failed(propagateMemScopeToUser(*this, val, memrefScope, user))) {
      return failure();
    }
  }
  return success();
}

LogicalResult
MemScopeInferAndPropagateHelper::Run(Value operand,
                                     const AddressSpaceAttr &targetMemScope) {
  auto memRefType = dyn_cast<BaseMemRefType>(operand.getType());
  if (!memRefType) {
    return failure();
  }

  auto memSpace = memRefType.getMemorySpace();
  if (memSpace) {
    if (memSpace != targetMemScope) {
      return failure();
    }
    return success();
  }

  // Update its scope.
  setBaseMemRefTypeScope(operand, targetMemScope);

  // Propagate the new memref type to its users.
  return propagateMemScopeToUsers(operand);
}

namespace {
struct InferPTOMemScopePass
    : public impl::InferPTOMemScopeBase<InferPTOMemScopePass> {
  void runOnOperation() override;

private:
  LogicalResult fixDeviceCallSite(func::FuncOp op);
  [[maybe_unused]] LogicalResult fixHostFuncSignature(func::FuncOp op);
};
} // namespace

template <typename OpT>
static LogicalResult ensureDpsOnlyOp(OpT op) {
  if (op.getNumResults() != 0) {
    return op->emitOpError(
        "Run infer memory scope after bufferization (Op must have 0 results).");
  }
  return success();
}

static AddressSpaceAttr getMemScopeAttr(MLIRContext *ctx,
                                        pto::AddressSpace scope) {
  return AddressSpaceAttr::get(ctx, scope);
}

static LogicalResult propagateOperandScopes(
    Operation *op,
    ArrayRef<std::tuple<Value, StringRef, AddressSpaceAttr>> specs) {
  MemScopeInferAndPropagateHelper helper;
  for (const auto &[value, valueName, targetScope] : specs) {
    if (failed(propagateAllocScope(op, value, valueName, targetScope, helper))) {
      return failure();
    }
  }
  return success();
}

LogicalResult pto::inferAndPropagateMemScopeForMovDps(pto::TMovOp op) {
  if (failed(ensureDpsOnlyOp(op))) {
    return failure();
  }

  auto dstAlloc = requireRootAlloc(op, op.getDst(), "mB");
  if (!dstAlloc.has_value()) {
    return failure();
  }

  auto memRefType = dyn_cast<BaseMemRefType>(dstAlloc->getType());
  if (!memRefType) {
    return op->emitOpError("Failed to infer/propagate memory scope for mA");
  }

  auto memSpace = memRefType.getMemorySpace();
  if (!memSpace) {
    return success();
  }

  auto l0aSpaceAttr =
      getMemScopeAttr(op->getContext(), pto::AddressSpace::LEFT);
  auto l0bSpaceAttr =
      getMemScopeAttr(op->getContext(), pto::AddressSpace::RIGHT);
  auto l0cSpaceAttr =
      getMemScopeAttr(op->getContext(), pto::AddressSpace::ACC);
  auto l1SpaceAttr = getMemScopeAttr(op->getContext(), pto::AddressSpace::MAT);
  auto ubSpaceAttr = getMemScopeAttr(op->getContext(), pto::AddressSpace::VEC);
  auto biasSpaceAttr =
      getMemScopeAttr(op->getContext(), pto::AddressSpace::BIAS);
  if (memSpace == ubSpaceAttr) {
    return propagateOperandScopes(op, {{op.getSrc(), "mA", ubSpaceAttr}});
  }
  if (memSpace == l1SpaceAttr) {
    return propagateOperandScopes(op, {{op.getSrc(), "mA", l0cSpaceAttr}});
  }
  if (memSpace == l0aSpaceAttr || memSpace == l0bSpaceAttr ||
      memSpace == biasSpaceAttr) {
    return propagateOperandScopes(op, {{op.getSrc(), "mA", l1SpaceAttr}});
  }
  return success();
}

LogicalResult pto::inferAndPropagateMemScopeForMatmulAccDps(pto::TMatmulAccOp op) {
  if (failed(ensureDpsOnlyOp(op))) {
    return failure();
  }

  return propagateOperandScopes(
      op, {{op.getAccIn(), "mAcc",
            getMemScopeAttr(op->getContext(), pto::AddressSpace::ACC)},
           {op.getLhs(), "mA",
            getMemScopeAttr(op->getContext(), pto::AddressSpace::LEFT)},
           {op.getRhs(), "mB",
            getMemScopeAttr(op->getContext(), pto::AddressSpace::RIGHT)},
           {op.getDst(), "mC",
            getMemScopeAttr(op->getContext(), pto::AddressSpace::ACC)}});
}


LogicalResult pto::inferAndPropagateMemScopeForMatmulBiasDps(pto::TMatmulBiasOp op) {
  if (failed(ensureDpsOnlyOp(op))) {
    return failure();
  }

  return propagateOperandScopes(
      op, {{op.getA(), "mA",
            getMemScopeAttr(op->getContext(), pto::AddressSpace::LEFT)},
           {op.getB(), "mB",
            getMemScopeAttr(op->getContext(), pto::AddressSpace::RIGHT)},
           {op.getDst(), "mC",
            getMemScopeAttr(op->getContext(), pto::AddressSpace::ACC)},
           {op.getBias(), "mD",
            getMemScopeAttr(op->getContext(), pto::AddressSpace::BIAS)}});
}

LogicalResult pto::inferAndPropagateMemScopeForMatmulDps(pto::TMatmulOp op) {
  if (failed(ensureDpsOnlyOp(op))) {
    return failure();
  }

  return propagateOperandScopes(
      op, {{op.getLhs(), "mA",
            getMemScopeAttr(op->getContext(), pto::AddressSpace::LEFT)},
           {op.getRhs(), "mB",
            getMemScopeAttr(op->getContext(), pto::AddressSpace::RIGHT)},
           {op.getDst(), "mC",
            getMemScopeAttr(op->getContext(), pto::AddressSpace::ACC)}});
}

LogicalResult InferPTOMemScopePass::fixDeviceCallSite(func::FuncOp op) {
  LDBG("Begin fixing call site for " << op.getSymName());
  MemScopeInferAndPropagateHelper helper;
  SymbolTable::UseRange uses = *op.getSymbolUses(getOperation());
  for (SymbolTable::SymbolUse use : uses) {
    func::CallOp call = cast<func::CallOp>(use.getUser());
    // propagate call operand's memory scope
    for (auto [idx, callOperand] : llvm::enumerate(call.getArgOperands())) {
      if (!isa<BaseMemRefType>(callOperand.getType())) {
        continue;
      }

      auto funcOperandType = op.getFunctionType().getInput(idx);
      if (!isa<BaseMemRefType>(funcOperandType)) {
        continue;
      }

      LDBG("call operand: " << callOperand);
      if (failed(helper.Run(tracebackMemRef(callOperand),
                            getPTOAddressSpaceAttr(funcOperandType)))) {
        return op->emitOpError()
               << "Failed to propagate memory scope for operand "
               << callOperand;
      }
      LDBG("call operand after: " << callOperand);
    }
    // propagate call return value memory scope
    for (auto [idx, returnValue] : llvm::enumerate(call->getResults())) {
      if (!isa<BaseMemRefType>(returnValue.getType())) {
        continue;
      }

      auto funcReturnType = op.getFunctionType().getResult(idx);
      if (!isa<BaseMemRefType>(funcReturnType)) {
        continue;
      }

      if (failed(helper.Run(returnValue,
                            getPTOAddressSpaceAttr(funcReturnType)))) {
        return op->emitOpError()
               << "Failed to propagate memory scope for result " << returnValue;
      }
    }
  }
  return success();
}

/// Update the function type for the host function.
/// Because we propagate information from the call site to the caller, we only
/// updated the memref type of the BlockArgument of or the return operation
/// within the function (if they are updated at all). So we need to use those
/// information to update the function's type.
[[maybe_unused]] LogicalResult InferPTOMemScopePass::fixHostFuncSignature(func::FuncOp op) {
  // Skip external host functions because we know nothing about it.
  if (op.isExternal()) {
    return success();
  }

  func::ReturnOp returnOp = getAssumedUniqueReturnOp(op);
  if (!returnOp) {
    return failure();
  }

  SmallVector<Type> newArgsType(llvm::map_to_vector(
      op.getArguments(), [](const BlockArgument &ba) { return ba.getType(); }));
  SmallVector<Type> newReturnType(llvm::map_to_vector(
      returnOp.getOperandTypes(), [](const Type &type) { return type; }));
  auto newFt = op.getFunctionType().clone(newArgsType, newReturnType);
  op.setFunctionType(newFt);
  return success();
}

static LogicalResult inferAndPropagateMemScopeForExternFunc(func::FuncOp op) {
  if (!op.isExternal()) {
    return failure();
  }

  auto gmSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::GM);
  LDBG("Begin infer and propagate memory scope for extern func"
       << op.getSymName());
  auto newArgTypes = SmallVector<Type>(op.getArgumentTypes());
  for (auto &argType : newArgTypes) {
    // If not base memref and already has memspace then skip
    if (auto memrefType = dyn_cast<BaseMemRefType>(argType)) {
      if (memrefType.getMemorySpace()) {
        continue;
      }
      argType = getBaseMemRefTypeWithNewScope(memrefType, gmSpaceAttr);
    }
  }
  // For extern functions that have results, we assume that the memory scope
  // is Global Memory.
  auto newReturnTypes = SmallVector<Type>(op.getResultTypes());
  for (auto &resultType : newReturnTypes) {
    // If not base memref and already has memspace then skip
    if (auto memrefType = dyn_cast<BaseMemRefType>(resultType)) {
      if (memrefType.getMemorySpace()) {
        continue;
      }
      resultType = getBaseMemRefTypeWithNewScope(memrefType, gmSpaceAttr);
    }
  }
  auto newFt = op.getFunctionType().clone(newArgTypes, newReturnTypes);
  op.setFunctionType(newFt);
  return success();
}

LogicalResult pto::inferAndPropagateMemScopeForFunc(func::FuncOp op) {
  if (op.isExternal()) {
    return inferAndPropagateMemScopeForExternFunc(op);
  }

  LDBG("Begin infer and propagate memory scope for func" << op.getSymName());
  MemScopeInferAndPropagateHelper helper;
  auto gmSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::GM);
  auto ubSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::VEC);
  auto args = op.getArguments();
  for (auto arg : args) {
    if (!isa<BaseMemRefType>(arg.getType())) {
      continue;
    }

    if (op->hasAttr(pto::VectorFunctionAttr::name)) {
      if (failed(helper.Run(arg, ubSpaceAttr))) {
        return op->emitOpError()
               << "Failed to propagate UB memory scope for argument # in VF"
               << arg.getArgNumber();
      }
    } else if (failed(helper.Run(arg, gmSpaceAttr))) {
      return op->emitOpError()
             << "Failed to propagate memory scope for argument #"
             << arg.getArgNumber();
    }
  }
  if (!args.empty()) {
    auto newFt = op.getFunctionType().clone(
        op.getBody().front().getArgumentTypes(), op.getResultTypes());
    op.setFunctionType(newFt);
  }
  if (op->getNumResults() > 0) {
    op.emitWarning()
        << "non-externl function has return value after bufferization!";
  }

  return success();
}

LogicalResult pto::inferAndPropagateMemScopeForGpuFunc(gpu::GPUFuncOp op) {
  MemScopeInferAndPropagateHelper helper;
  auto gmSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::GM);

  auto args = op.getArguments();
  for (auto arg : args) {
    if (!isa<BaseMemRefType>(arg.getType())) {
      continue;
    }

    // GPUFuncOp arguments are currently treated as GM unless a caller-provided
    // scope overrides them.
    if (failed(helper.Run(arg, gmSpaceAttr))) {
      return op->emitOpError()
             << "Failed to propagate memory scope for argument #"
             << arg.getArgNumber();
    }
  }

  if (!args.empty()) {
    auto newFt = op.getFunctionType().clone(
        op.getBody().front().getArgumentTypes(), op.getResultTypes());
    op.setFunctionType(newFt);
  }

  return success();
}

LogicalResult pto::inferAndPropagateUbufMemScope(memref::AllocOp op) {
  LDBG("Begin infer and propagate memory scope for: " << *op);
  auto memorySpace = op.getType().getMemorySpace();
  if (memorySpace) {
    return success();
  }

  MemScopeInferAndPropagateHelper helper;
  auto ubSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::VEC);
  if (failed(helper.Run(op, ubSpaceAttr))) {
    return op->emitOpError("Failed to propagate memory scope ub for allocOp");
  }
  return success();
}

void InferPTOMemScopePass::runOnOperation() {
  SmallVector<func::FuncOp> deviceFuncList;
  getOperation()->walk([&](func::FuncOp func) {
    deviceFuncList.push_back(func);
    return;
  });

  SmallVector<gpu::GPUFuncOp> gpuFuncList;
  getOperation()->walk([&](gpu::GPUModuleOp gpuModule) {
    gpuModule->walk([&](gpu::GPUFuncOp gpuFunc) -> void {
      gpuFuncList.push_back(gpuFunc);
    });
  });

  for (auto func : gpuFuncList) {
    if (failed(inferAndPropagateMemScopeForGpuFunc(func))) {
      signalPassFailure();
    }
  }

  // Infer and propagate memory scope for device functions.
  for (auto func : deviceFuncList) {
    // Set the memory scope of values related to `pto::MmadL1Op` to L1 or L0C.
    func->walk([&](mlir::pto::TMatmulOp op) {
      if (failed(pto::inferAndPropagateMemScopeForMatmulDps(op))) {
        signalPassFailure();
      }
    });

    func->walk([&](mlir::pto::TMatmulAccOp op) {
      if (failed(pto::inferAndPropagateMemScopeForMatmulAccDps(op))) {
        signalPassFailure();
      }
    });

    func->walk([&](mlir::pto::TMatmulBiasOp op) {
      if (failed(pto::inferAndPropagateMemScopeForMatmulBiasDps(op))) {
        signalPassFailure();
      }
    });

    func->walk([&](mlir::pto::TMovOp op) {
      if (failed(pto::inferAndPropagateMemScopeForMovDps(op))) {
        signalPassFailure();
      }
    });

    // Set device function arguments' memory scope to GM.
    if (failed(pto::inferAndPropagateMemScopeForFunc(func))) {
      signalPassFailure();
    }

    // Finally, set the remaining memory scope in the device kernel to UB.
    func->walk([&](memref::AllocOp op) {
      if (failed(pto::inferAndPropagateUbufMemScope(op))) {
        signalPassFailure();
      }
    });
  }

  for (auto func : deviceFuncList) {
    if (failed(fixDeviceCallSite(func))) {
      signalPassFailure();
    }
  }
}

std::unique_ptr<Pass> mlir::pto::createInferPTOMemScopePass() {
  return std::make_unique<InferPTOMemScopePass>();
}

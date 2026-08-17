// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIMaskGranularityAssignment.cpp - Assign VMI mask granularity -----===//
//===----------------------------------------------------------------------===//
//
// This pass assigns concrete b8/b16/b32 granularity to VMI mask values before
// layout assignment.  It deliberately does not choose layouts: mask layout is
// assigned later by vmi-layout-assignment.  When a mask value has conflicting
// granularity uses, this pass keeps the value's primary granularity and either
// rematerializes cheap mask producers at the use site or inserts
// pto.vmi.ensure_mask_granularity.

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMIControlFlowSupport.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMIMASKGRANULARITYASSIGNMENT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kMaskGranularity8Bit = 8;
constexpr unsigned kMaskGranularity16Bit = 16;
constexpr unsigned kMaskGranularity32Bit = 32;
constexpr unsigned kInvalidMaskId = ~0U;

struct MaskNode {
  Value value;
  VMIMaskType type;
  unsigned parent = 0;
  std::string granularity;
};

struct MaskUseRequest {
  OpOperand *operand;
  std::string granularity;
};

static unsigned getElementBitWidth(Type type) {
  if (isa<IndexType>(type)) {
    return mlir::pto::kValue64;
  }
  return pto::getPTOStorageElemBitWidth(type);
}

static StringRef getMaskGranularityForElement(Type elementType) {
  switch (getElementBitWidth(elementType)) {
  case kMaskGranularity8Bit:
    return "b8";
  case kMaskGranularity16Bit:
    return "b16";
  case kMaskGranularity32Bit:
    return "b32";
  default:
    return "";
  }
}

static bool containsVMIType(Type type) {
  if (isa<VMIVRegType, VMIMaskType>(type)) {
    return true;
  }
  if (auto functionType = dyn_cast<FunctionType>(type)) {
    return llvm::any_of(functionType.getInputs(), containsVMIType) ||
           llvm::any_of(functionType.getResults(), containsVMIType);
  }
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return containsVMIType(shapedType.getElementType());
  }
  return false;
}

struct MaskGranularitySolver {
  explicit MaskGranularitySolver(ModuleOp module)
      : module(module), ctx(module.getContext()) {}

  unsigned addMaskValue(Value value) {
    auto type = dyn_cast<VMIMaskType>(value.getType());
    if (!type) {
      return kInvalidMaskId;
    }
    auto [it, inserted] = maskIds.try_emplace(value, maskNodes.size());
    if (inserted) {
      std::string granularity;
      if (VMIMaskType::isConcreteGranularity(type.getGranularity())) {
        granularity = type.getGranularity().str();
      }
      maskNodes.push_back(MaskNode{value, type, it->second, granularity});
    }
    return it->second;
  }

  unsigned findMask(unsigned id) {
    if (maskNodes[id].parent == id) {
      return id;
    }
    maskNodes[id].parent = findMask(maskNodes[id].parent);
    return maskNodes[id].parent;
  }

  LogicalResult uniteMask(Value lhs, Value rhs, Operation *op) {
    unsigned lhsId = addMaskValue(lhs);
    unsigned rhsId = addMaskValue(rhs);
    if (lhsId == kInvalidMaskId || rhsId == kInvalidMaskId) {
      return success();
    }
    unsigned lhsRoot = findMask(lhsId);
    unsigned rhsRoot = findMask(rhsId);
    if (lhsRoot == rhsRoot) {
      return success();
    }

    MaskNode &lhsNode = maskNodes[lhsRoot];
    MaskNode &rhsNode = maskNodes[rhsRoot];
    if (!lhsNode.granularity.empty() && !rhsNode.granularity.empty() &&
        lhsNode.granularity != rhsNode.granularity) {
      return op->emitError() << kVMIDiagLayoutContractPrefix
                             << "conflicting mask granularities "
                             << lhsNode.granularity << " and "
                             << rhsNode.granularity;
    }

    rhsNode.parent = lhsRoot;
    if (lhsNode.granularity.empty()) {
      lhsNode.granularity = rhsNode.granularity;
    }
    return success();
  }

  LogicalResult requestMask(Value mask, StringRef granularity, Operation *op) {
    unsigned id = addMaskValue(mask);
    if (id == ~0U) {
      return success();
    }
    if (granularity.empty()) {
      return op->emitError() << kVMIDiagLayoutContractPrefix
                             << "cannot infer concrete mask granularity";
    }
    MaskNode &node = maskNodes[findMask(id)];
    if (!node.granularity.empty() && node.granularity != granularity) {
      return op->emitError()
             << kVMIDiagLayoutContractPrefix
             << "conflicting mask granularities " << node.granularity << " and "
             << granularity;
    }
    node.granularity = granularity.str();
    return success();
  }

  LogicalResult requestMaskUse(OpOperand &operand, StringRef granularity,
                               Operation *op) {
    if (!isa<VMIMaskType>(operand.get().getType())) {
      return success();
    }
    if (granularity.empty()) {
      return op->emitError() << kVMIDiagLayoutContractPrefix
                             << "cannot infer concrete mask use granularity";
    }
    maskUseRequests.push_back(MaskUseRequest{&operand, granularity.str()});
    return success();
  }

  LogicalResult collect() {
    module.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        addMaskValue(result);
      }
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments()) {
            addMaskValue(arg);
          }
        }
      }
    });
    return success();
  }

  LogicalResult addConstraints() {
    WalkResult result = module.walk([&](Operation *op) -> WalkResult {
      if (auto maskAnd = dyn_cast<VMIMaskAndOp>(op)) {
        if (failed(uniteMask(maskAnd.getLhs(), maskAnd.getRhs(), op)) ||
            failed(uniteMask(maskAnd.getLhs(), maskAnd.getResult(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto maskOr = dyn_cast<VMIMaskOrOp>(op)) {
        if (failed(uniteMask(maskOr.getLhs(), maskOr.getRhs(), op)) ||
            failed(uniteMask(maskOr.getLhs(), maskOr.getResult(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto maskXor = dyn_cast<VMIMaskXOrOp>(op)) {
        if (failed(uniteMask(maskXor.getLhs(), maskXor.getRhs(), op)) ||
            failed(uniteMask(maskXor.getLhs(), maskXor.getResult(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto maskNot = dyn_cast<VMIMaskNotOp>(op)) {
        if (failed(uniteMask(maskNot.getSource(), maskNot.getResult(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto ensure = dyn_cast<VMIEnsureMaskLayoutOp>(op)) {
        if (failed(uniteMask(ensure.getSource(), ensure.getResult(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto cmpf = dyn_cast<VMICmpFOp>(op)) {
        auto lhsType = cast<VMIVRegType>(cmpf.getLhs().getType());
        if (failed(requestMask(
                cmpf.getResult(),
                getMaskGranularityForElement(lhsType.getElementType()), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto cmpi = dyn_cast<VMICmpIOp>(op)) {
        auto lhsType = cast<VMIVRegType>(cmpi.getLhs().getType());
        if (failed(requestMask(
                cmpi.getResult(),
                getMaskGranularityForElement(lhsType.getElementType()), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto select = dyn_cast<VMISelectOp>(op)) {
        auto resultType = cast<VMIVRegType>(select.getResult().getType());
        if (failed(requestMaskUse(
                select.getMaskMutable(),
                getMaskGranularityForElement(resultType.getElementType()), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto vecScalar = dyn_cast<VMIAddSOp>(op)) {
        if (failed(requestMaskUseForSource(vecScalar.getMaskMutable(),
                                           vecScalar.getSrc(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto vecScalar = dyn_cast<VMIMulSOp>(op)) {
        if (failed(requestMaskUseForSource(vecScalar.getMaskMutable(),
                                           vecScalar.getSrc(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto vecScalar = dyn_cast<VMIMaxSOp>(op)) {
        if (failed(requestMaskUseForSource(vecScalar.getMaskMutable(),
                                           vecScalar.getSrc(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto vecScalar = dyn_cast<VMIMinSOp>(op)) {
        if (failed(requestMaskUseForSource(vecScalar.getMaskMutable(),
                                           vecScalar.getSrc(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto vecScalar = dyn_cast<VMIShlSOp>(op)) {
        if (failed(requestMaskUseForSource(vecScalar.getMaskMutable(),
                                           vecScalar.getSrc(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto vecScalar = dyn_cast<VMIShrSOp>(op)) {
        if (failed(requestMaskUseForSource(vecScalar.getMaskMutable(),
                                           vecScalar.getSrc(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto vexpdif = dyn_cast<VMIVexpdifOp>(op)) {
        if (failed(requestMaskUseForSource(vexpdif.getMaskMutable(),
                                           vexpdif.getX(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto vmull = dyn_cast<VMIVmullOp>(op)) {
        if (failed(requestMaskUse(vmull.getMaskMutable(), "b32", op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto addc = dyn_cast<VMIVaddcOp>(op)) {
        if (failed(requestMaskUse(addc.getMaskMutable(), "b32", op)) ||
            failed(requestMask(addc.getCarry(), "b32", op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto addcs = dyn_cast<VMIVaddcsOp>(op)) {
        if (failed(requestMaskUse(addcs.getCarryInMutable(), "b32", op)) ||
            failed(requestMaskUse(addcs.getMaskMutable(), "b32", op)) ||
            failed(requestMask(addcs.getCarry(), "b32", op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto activePrefix = dyn_cast<VMIActivePrefixIndexOp>(op)) {
        auto resultType = cast<VMIVRegType>(activePrefix.getResult().getType());
        if (failed(requestMaskUse(
                activePrefix.getMaskMutable(),
                getMaskGranularityForElement(resultType.getElementType()), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto compress = dyn_cast<VMICompressOp>(op)) {
        auto resultType = cast<VMIVRegType>(compress.getResult().getType());
        if (failed(requestMaskUse(
                compress.getMaskMutable(),
                getMaskGranularityForElement(resultType.getElementType()), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto vintlv = dyn_cast<VMIVintlvOp>(op)) {
        if (failed(requestMaskUseForSource(vintlv.getMaskMutable(),
                                           vintlv.getLhs(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto vdintlv = dyn_cast<VMIVdintlvOp>(op)) {
        if (failed(requestMaskUseForSource(vdintlv.getMaskMutable(),
                                           vdintlv.getLhs(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceAddIOp>(op)) {
        if (failed(requestMaskUseForSource(reduce.getMaskMutable(),
                                           reduce.getSource(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceAddFOp>(op)) {
        if (failed(requestMaskUseForSource(reduce.getMaskMutable(),
                                           reduce.getSource(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceMaxFOp>(op)) {
        if (failed(requestMaskUseForSource(reduce.getMaskMutable(),
                                           reduce.getSource(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceMinFOp>(op)) {
        if (failed(requestMaskUseForSource(reduce.getMaskMutable(),
                                           reduce.getSource(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceMaxIOp>(op)) {
        if (failed(requestMaskUseForSource(reduce.getMaskMutable(),
                                           reduce.getSource(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceMinIOp>(op)) {
        if (failed(requestMaskUseForSource(reduce.getMaskMutable(),
                                           reduce.getSource(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceAddFOp>(op)) {
        if (failed(requestMaskUseForSource(reduce.getMaskMutable(),
                                           reduce.getSource(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceMaxFOp>(op)) {
        if (failed(requestMaskUseForSource(reduce.getMaskMutable(),
                                           reduce.getSource(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceMinFOp>(op)) {
        if (failed(requestMaskUseForSource(reduce.getMaskMutable(),
                                           reduce.getSource(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceAddIOp>(op)) {
        if (failed(requestMaskUseForSource(reduce.getMaskMutable(),
                                           reduce.getSource(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceMaxIOp>(op)) {
        if (failed(requestMaskUseForSource(reduce.getMaskMutable(),
                                           reduce.getSource(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceMinIOp>(op)) {
        if (failed(requestMaskUseForSource(reduce.getMaskMutable(),
                                           reduce.getSource(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto hist = dyn_cast<VMIVdhistOp>(op)) {
        if (failed(requestMaskUse(hist.getMaskMutable(), "b8", op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto hist = dyn_cast<VMIVchistOp>(op)) {
        if (failed(requestMaskUse(hist.getMaskMutable(), "b8", op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIStrideLoadOp>(op)) {
        auto resultType = cast<VMIVRegType>(load.getResult().getType());
        if (failed(requestMaskUse(
                load.getMaskMutable(),
                getMaskGranularityForElement(resultType.getElementType()), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIMaskedLoadOp>(op)) {
        auto resultType = cast<VMIVRegType>(load.getResult().getType());
        if (failed(requestMaskUse(
                load.getMaskMutable(),
                getMaskGranularityForElement(resultType.getElementType()), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto gather = dyn_cast<VMIGatherOp>(op)) {
        auto resultType = cast<VMIVRegType>(gather.getResult().getType());
        if (failed(requestMaskUse(
                gather.getMaskMutable(),
                getMaskGranularityForElement(resultType.getElementType()), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIExpandLoadOp>(op)) {
        auto resultType = cast<VMIVRegType>(load.getResult().getType());
        if (failed(requestMaskUse(
                load.getMaskMutable(),
                getMaskGranularityForElement(resultType.getElementType()), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMIMaskedStoreOp>(op)) {
        if (failed(requestMaskUseForSource(store.getMaskMutable(),
                                           store.getValue(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMIStrideStoreOp>(op)) {
        if (failed(requestMaskUseForSource(store.getMaskMutable(),
                                           store.getValue(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto scatter = dyn_cast<VMIScatterOp>(op)) {
        if (failed(requestMaskUseForSource(scatter.getMaskMutable(),
                                           scatter.getValue(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMICompressStoreOp>(op)) {
        if (failed(requestMaskUseForSource(store.getMaskMutable(),
                                           store.getValue(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (failed(addIfConstraints(ifOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto executeOp = dyn_cast<scf::ExecuteRegionOp>(op)) {
        if (failed(addExecuteRegionConstraints(executeOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto indexSwitchOp = dyn_cast<scf::IndexSwitchOp>(op)) {
        if (failed(addIndexSwitchConstraints(indexSwitchOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
        if (failed(addWhileConstraints(whileOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (failed(addForConstraints(forOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto branchOp = dyn_cast<cf::BranchOp>(op)) {
        if (failed(addBranchConstraints(branchOp.getDest(),
                                        branchOp.getDestOperands(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(op)) {
        if (failed(addBranchConstraints(condBranchOp.getTrueDest(),
                                        condBranchOp.getTrueDestOperands(),
                                        op)) ||
            failed(addBranchConstraints(condBranchOp.getFalseDest(),
                                        condBranchOp.getFalseOperands(), op))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto switchOp = dyn_cast<cf::SwitchOp>(op)) {
        if (failed(addBranchConstraints(switchOp.getDefaultDestination(),
                                        switchOp.getDefaultOperands(), op))) {
          return WalkResult::interrupt();
        }
        for (auto [dest, operands] : llvm::zip(switchOp.getCaseDestinations(),
                                               switchOp.getCaseOperands())) {
          if (failed(addBranchConstraints(dest, operands, op))) {
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      }
      if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
        if (failed(addReturnConstraints(returnOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        if (failed(addCallConstraints(callOp))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (op->getName().getStringRef() == "func.call_indirect" &&
          hasVMIValueTypes(op)) {
        op->emitError() << kVMIDiagLayoutContractPrefix
                        << "VMI typed call requires a direct internal callee "
                           "with a body";
        return WalkResult::interrupt();
      }
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (funcOp.empty() && hasVMIFunctionType(funcOp)) {
          funcOp.emitError()
              << kVMIDiagLayoutContractPrefix
              << "VMI typed function declaration requires an explicit "
                 "external ABI materialization plan";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult requestMaskUseForSource(OpOperand &mask, Value source,
                                        Operation *op) {
    auto sourceType = dyn_cast<VMIVRegType>(source.getType());
    if (!sourceType) {
      return success();
    }
    return requestMaskUse(mask,
                          getMaskGranularityForElement(
                              sourceType.getElementType()),
                          op);
  }

  LogicalResult uniteEquivalentValues(Value lhs, Value rhs, Operation *op) {
    return uniteMask(lhs, rhs, op);
  }

  LogicalResult addIfConstraints(scf::IfOp ifOp) {
    for (OpResult result : ifOp->getResults()) {
      unsigned resultNo = result.getResultNumber();
      for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
        if (region->empty()) {
          continue;
        }
        auto yieldOp = dyn_cast<scf::YieldOp>(region->front().getTerminator());
        if (!yieldOp || resultNo >= yieldOp.getNumOperands()) {
          continue;
        }
        if (failed(uniteEquivalentValues(result, yieldOp.getOperand(resultNo),
                                         ifOp))) {
          return failure();
        }
      }
    }
    return success();
  }

  LogicalResult addYieldConstraints(ResultRange results, scf::YieldOp yieldOp,
                                    Operation *op) {
    for (auto [index, result] : llvm::enumerate(results)) {
      if (index >= yieldOp.getNumOperands()) {
        break;
      }
      if (failed(uniteEquivalentValues(result, yieldOp.getOperand(index), op))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addExecuteRegionConstraints(scf::ExecuteRegionOp executeOp) {
    WalkResult result = executeOp.getRegion().walk([&](scf::YieldOp yieldOp) {
      if (yieldOp->getParentOp() != executeOp.getOperation()) {
        return WalkResult::advance();
      }
      if (failed(
              addYieldConstraints(executeOp->getResults(), yieldOp, executeOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult addIndexSwitchConstraints(scf::IndexSwitchOp indexSwitchOp) {
    auto addBlockTerminator = [&](Block &block) -> LogicalResult {
      auto yieldOp = dyn_cast<scf::YieldOp>(block.getTerminator());
      if (!yieldOp) {
        return success();
      }
      return addYieldConstraints(indexSwitchOp->getResults(), yieldOp,
                                 indexSwitchOp);
    };
    if (failed(addBlockTerminator(indexSwitchOp.getDefaultBlock()))) {
      return failure();
    }
    for (unsigned idx = 0, e = indexSwitchOp.getNumCases(); idx < e; ++idx) {
      if (failed(addBlockTerminator(indexSwitchOp.getCaseBlock(idx)))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addWhileConstraints(scf::WhileOp whileOp) {
    return VMIControlFlowSupport::addWhileConstraints(
        whileOp, [&](Value lhs, Value rhs, Operation *op) {
          return uniteEquivalentValues(lhs, rhs, op);
        });
  }

  LogicalResult addForConstraints(scf::ForOp forOp) {
    return VMIControlFlowSupport::addForConstraints(
        forOp, [&](Value lhs, Value rhs, Operation *op) {
          return uniteEquivalentValues(lhs, rhs, op);
        });
  }

  LogicalResult addBranchConstraints(Block *dest, OperandRange operands,
                                     Operation *op) {
    if (!dest) {
      return success();
    }
    for (auto [index, operand] : llvm::enumerate(operands)) {
      if (index >= dest->getNumArguments()) {
        break;
      }
      if (failed(uniteEquivalentValues(operand, dest->getArgument(index), op))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addReturnConstraints(func::ReturnOp returnOp) {
    auto func = returnOp->getParentOfType<func::FuncOp>();
    if (!func) {
      return success();
    }

    auto it = firstReturnOperandsByFunc.find(func);
    if (it == firstReturnOperandsByFunc.end()) {
      SmallVector<Value> operands(returnOp.getOperands());
      firstReturnOperandsByFunc.try_emplace(func, std::move(operands));
      return success();
    }

    ArrayRef<Value> firstOperands = it->second;
    for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
      if (index >= firstOperands.size()) {
        break;
      }
      if (failed(
              uniteEquivalentValues(firstOperands[index], operand, returnOp))) {
        return failure();
      }
    }
    return success();
  }

  bool hasVMIValueTypes(Operation *op) {
    return llvm::any_of(op->getOperandTypes(), containsVMIType) ||
           llvm::any_of(op->getResultTypes(), containsVMIType);
  }

  bool hasVMIFunctionType(func::FuncOp func) {
    FunctionType type = func.getFunctionType();
    return llvm::any_of(type.getInputs(), containsVMIType) ||
           llvm::any_of(type.getResults(), containsVMIType);
  }

  LogicalResult addCallConstraints(func::CallOp callOp) {
    if (!hasVMIValueTypes(callOp)) {
      return success();
    }

    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        callOp, callOp.getCalleeAttr());
    if (!callee || callee.empty()) {
      return callOp.emitError()
             << kVMIDiagLayoutContractPrefix
             << "VMI typed call requires a direct internal callee with a body";
    }

    for (auto [operand, argument] :
         llvm::zip(callOp.getOperands(), callee.getArguments())) {
      if (failed(uniteEquivalentValues(operand, argument, callOp))) {
        return failure();
      }
    }

    SmallVector<func::ReturnOp> returns;
    callee.walk([&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
    for (func::ReturnOp returnOp : returns) {
      for (auto [index, result] : llvm::enumerate(callOp.getResults())) {
        if (index >= returnOp.getNumOperands()) {
          break;
        }
        if (failed(uniteEquivalentValues(result, returnOp.getOperand(index),
                                         callOp))) {
          return failure();
        }
      }
    }
    return success();
  }

  void rewriteMaskTypes() {
    for (MaskNode &node : maskNodes) {
      MaskNode &root = maskNodes[findMask(maskIds.lookup(node.value))];
      StringRef granularity =
          root.granularity.empty() ? StringRef("b32") : StringRef(root.granularity);
      node.value.setType(VMIMaskType::get(ctx, node.type.getElementCount(),
                                          granularity,
                                          node.type.getLayoutAttr()));
    }
  }

  SmallVector<Type> getCallResultTypes(func::FuncOp func) {
    SmallVector<Type> resultTypes;
    bool found = false;
    module.walk([&](func::CallOp call) {
      if (call.getCallee() != func.getSymName()) {
        return;
      }
      if (!found) {
        resultTypes.assign(call.getResultTypes().begin(),
                           call.getResultTypes().end());
        found = true;
        return;
      }
      if (resultTypes.size() != call.getNumResults()) {
        return;
      }
      for (auto [index, type] : llvm::enumerate(call.getResultTypes())) {
        if (index < resultTypes.size() && resultTypes[index] != type) {
          resultTypes[index] = {};
        }
      }
    });
    return found ? resultTypes : SmallVector<Type>{};
  }

  void rewriteFunctionType() {
    module.walk([&](func::FuncOp func) {
      if (func.empty()) {
        return;
      }

      SmallVector<Type> inputs;
      inputs.reserve(func.getNumArguments());
      for (BlockArgument arg : func.getArguments()) {
        inputs.push_back(arg.getType());
      }

      SmallVector<Type> results;
      SmallVector<Type> callResultTypes = getCallResultTypes(func);
      auto it = firstReturnOperandsByFunc.find(func);
      if (!callResultTypes.empty()) {
        for (Type type : callResultTypes) {
          results.push_back(type ? type : Type{});
        }
      } else if (it != firstReturnOperandsByFunc.end()) {
        for (Value operand : it->second) {
          results.push_back(operand.getType());
        }
      } else {
        FunctionType functionType = func.getFunctionType();
        for (Type type : functionType.getResults()) {
          if (auto maskType = dyn_cast<VMIMaskType>(type)) {
            StringRef granularity =
                VMIMaskType::isConcreteGranularity(maskType.getGranularity())
                    ? maskType.getGranularity()
                    : StringRef("b32");
            results.push_back(VMIMaskType::get(
                ctx, maskType.getElementCount(), granularity,
                maskType.getLayoutAttr()));
          } else {
            results.push_back(type);
          }
        }
      }

      for (auto [index, type] : llvm::enumerate(results)) {
        if (!type) {
          results[index] = func.getFunctionType().getResult(index);
        }
      }

      func.setFunctionType(FunctionType::get(ctx, inputs, results));
    });
  }

  LogicalResult insertMaskUseMaterializations() {
    OpBuilder builder(ctx);
    for (MaskUseRequest request : maskUseRequests) {
      Value value = request.operand->get();
      auto sourceType = dyn_cast<VMIMaskType>(value.getType());
      if (!sourceType) {
        continue;
      }
      if (sourceType.getGranularity() == request.granularity) {
        continue;
      }

      builder.setInsertionPoint(request.operand->getOwner());
      auto resultType = VMIMaskType::get(ctx, sourceType.getElementCount(),
                                         request.granularity,
                                         sourceType.getLayoutAttr());
      Value current = rematerializeMaskProducer(
          value, resultType, request.operand->getOwner()->getLoc(), builder);
      if (!current) {
        current = builder.create<VMIEnsureMaskGranularityOp>(
            request.operand->getOwner()->getLoc(), resultType, value);
      }
      request.operand->set(current);
    }
    return success();
  }

  Value rematerializeMaskProducer(Value value, VMIMaskType resultType,
                                  Location loc, OpBuilder &builder) {
    if (auto createMask = value.getDefiningOp<VMICreateMaskOp>()) {
      return builder
          .create<VMICreateMaskOp>(loc, resultType, createMask.getActiveLanes())
          .getResult();
    }

    if (auto createGroupMask = value.getDefiningOp<VMICreateGroupMaskOp>()) {
      return builder
          .create<VMICreateGroupMaskOp>(
              loc, resultType, createGroupMask.getActiveElemsPerGroup(),
              createGroupMask.getNumGroupsAttr(),
              createGroupMask.getGroupSizeAttr())
          .getResult();
    }

    if (auto constantMask = value.getDefiningOp<VMIConstantMaskOp>()) {
      return builder
          .create<VMIConstantMaskOp>(loc, resultType,
                                     constantMask.getValueAttr())
          .getResult();
    }

    return {};
  }

  LogicalResult run() {
    if (failed(collect())) {
      return failure();
    }
    if (failed(addConstraints())) {
      return failure();
    }
    rewriteMaskTypes();
    rewriteFunctionType();
    return insertMaskUseMaterializations();
  }

  ModuleOp module;
  MLIRContext *ctx;
  DenseMap<Value, unsigned> maskIds;
  DenseMap<func::FuncOp, SmallVector<Value>> firstReturnOperandsByFunc;
  SmallVector<MaskNode> maskNodes;
  SmallVector<MaskUseRequest> maskUseRequests;
};

struct VMIMaskGranularityAssignmentPass
    : public mlir::pto::impl::VMIMaskGranularityAssignmentBase<
          VMIMaskGranularityAssignmentPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMIMaskGranularityAssignmentPass)

  void runOnOperation() override {
    if (failed(MaskGranularitySolver(getOperation()).run())) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMIMaskGranularityAssignmentPass() {
  return std::make_unique<VMIMaskGranularityAssignmentPass>();
}

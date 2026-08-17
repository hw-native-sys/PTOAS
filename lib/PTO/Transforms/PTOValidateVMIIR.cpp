// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOValidateVMIIR.cpp - VMI boundary verifier ----------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMILayoutSupport.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVALIDATEVMIIR
#define GEN_PASS_DEF_PTOVALIDATEVMILAYOUTIR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

bool isVMIType(Type type) { return isa<VMIVRegType, VMIMaskType>(type); }

bool containsVMIType(Type type) {
  if (isVMIType(type)) {
    return true;
  }

  if (auto functionType = dyn_cast<FunctionType>(type)) {
    return llvm::any_of(functionType.getInputs(),
                        [](Type input) { return containsVMIType(input); }) ||
           llvm::any_of(functionType.getResults(), [](Type result) {
             return containsVMIType(result);
           });
  }

  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return containsVMIType(shapedType.getElementType());
  }

  return false;
}

bool containsVMIType(Attribute attr) {
  if (!attr) {
    return false;
  }

  if (auto typeAttr = dyn_cast<TypeAttr>(attr)) {
    if (containsVMIType(typeAttr.getValue())) {
      return true;
    }
  }

  if (auto typedAttr = dyn_cast<TypedAttr>(attr)) {
    if (containsVMIType(typedAttr.getType())) {
      return true;
    }
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    return llvm::any_of(arrayAttr, [](Attribute element) {
      return containsVMIType(element);
    });
  }

  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
    return llvm::any_of(dictAttr, [](NamedAttribute namedAttr) {
      return containsVMIType(namedAttr.getValue());
    });
  }

  return false;
}

bool isSurfaceVMIType(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    return !vregType.getLayout();
  }
  if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    return maskType.isPred() && !maskType.getLayout();
  }
  return false;
}

bool isLayoutAssignedVMIType(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    return static_cast<bool>(vregType.getLayoutAttr());
  }
  if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    return maskType.getLayoutAttr() &&
           VMIMaskType::isConcreteGranularity(maskType.getGranularity());
  }
  return false;
}

bool isVMIHelperOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "pto.vmi.ensure_layout" ||
         name == "pto.vmi.ensure_mask_layout" ||
         name == "pto.vmi.ensure_mask_granularity" || name == "pto.vmi.pack" ||
         name == "pto.vmi.unpack";
}

bool isVMILayoutHelperOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "pto.vmi.ensure_layout" ||
         name == "pto.vmi.ensure_mask_layout" ||
         name == "pto.vmi.ensure_mask_granularity";
}

bool isVMISemanticOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name.starts_with("pto.vmi.") && !isVMIHelperOp(op);
}

bool isStructuralOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "builtin.module" || name.starts_with("func.") ||
         name.starts_with("scf.") || name.starts_with("cf.");
}

bool hasVMIType(Operation *op) {
  if (llvm::any_of(op->getOperandTypes(), isVMIType) ||
      llvm::any_of(op->getResultTypes(), isVMIType)) {
    return true;
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      if (llvm::any_of(block.getArgumentTypes(), isVMIType)) {
        return true;
      }
    }
  }
  return false;
}

void mirrorDiagnostic(llvm::raw_ostream *diagOS, Twine message) {
  if (diagOS) {
    *diagOS << message << "\n";
  }
}

LogicalResult emitInvariant(Operation *op, llvm::raw_ostream *diagOS,
                            Twine message) {
  InFlightDiagnostic diag = op->emitError()
                            << kVMIDiagPassInvariantPrefix << message;
  (void)diag;
  mirrorDiagnostic(diagOS, Twine(kVMIDiagPassInvariantPrefix) + message);
  return failure();
}

LogicalResult emitLayoutContract(Operation *op, llvm::raw_ostream *diagOS,
                                 Twine message) {
  InFlightDiagnostic diag = op->emitError()
                            << kVMIDiagLayoutContractPrefix << message;
  (void)diag;
  mirrorDiagnostic(diagOS, Twine(kVMIDiagLayoutContractPrefix) + message);
  return failure();
}

LogicalResult emitLayoutSupportContract(Operation *op,
                                        llvm::raw_ostream *diagOS,
                                        Twine message, StringRef reason) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << message << ": " << reason;

  bool printedAny = false;
  auto printValueType = [&os, &printedAny](StringRef kind, int64_t index,
                                            Type type) {
    if (!isVMIType(type)) {
      return;
    }
    if (!printedAny) {
      os << "; VMI types:";
      printedAny = true;
    }
    os << " " << kind << "#" << index << "=" << type;
  };

  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    printValueType("operand", static_cast<int64_t>(index), operand.getType());
  }
  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    printValueType("result", static_cast<int64_t>(index), result.getType());
  }

  os.flush();
  return emitLayoutContract(op, diagOS, text);
}

LogicalResult
emitHelperMaterializationContract(Operation *helper, Type sourceType,
                                  Type resultType, StringRef helperName,
                                  StringRef reason, llvm::raw_ostream *diagOS) {
  auto emitFallback = [diagOS, helper, helperName, reason]() {
    return emitLayoutContract(
        helper, diagOS,
        Twine(helperName) +
            " has no registered materialization support: " + reason);
  };

  if (helper->getNumResults() != 1 || !helper->getResult(0).hasOneUse()) {
    return emitFallback();
  }

  OpOperand &use = *helper->getResult(0).use_begin();
  Operation *requester = use.getOwner();
  std::string message;
  llvm::raw_string_ostream os(message);
  os << requester->getName() << " operand #" << use.getOperandNumber()
     << " has type " << sourceType << " but requires " << resultType << "; "
     << helperName << " has no registered materialization support: " << reason;
  os.flush();

  InFlightDiagnostic diag = requester->emitError()
                            << kVMIDiagLayoutContractPrefix << message;
  diag.attachNote(helper->getLoc())
      << "failed helper conversion " << sourceType << " -> " << resultType
      << " (" << reason << ")";
  mirrorDiagnostic(diagOS, Twine(kVMIDiagLayoutContractPrefix) + message);
  return failure();
}

LogicalResult verifyBoundaryType(Operation *owner, Type type,
                                 llvm::raw_ostream *diagOS) {
  if (isVMIType(type) && !isSurfaceVMIType(type)) {
    return emitInvariant(
        owner, diagOS,
        "VMI producer boundary requires surface !pto.vmi.vreg or "
        "!pto.vmi.mask<Nxpred> type");
  }

  return success();
}

LogicalResult verifyBoundaryTypeTree(Operation *owner, Type type,
                                     llvm::raw_ostream *diagOS) {
  if (failed(verifyBoundaryType(owner, type, diagOS))) {
    return failure();
  }

  if (auto functionType = dyn_cast<FunctionType>(type)) {
    for (Type input : functionType.getInputs()) {
      if (failed(verifyBoundaryTypeTree(owner, input, diagOS))) {
        return failure();
      }
    }
    for (Type result : functionType.getResults()) {
      if (failed(verifyBoundaryTypeTree(owner, result, diagOS))) {
        return failure();
      }
    }
  }

  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return verifyBoundaryTypeTree(owner, shapedType.getElementType(), diagOS);
  }

  return success();
}

LogicalResult verifyLayoutAssignedType(Operation *owner, Type type,
                                       llvm::raw_ostream *diagOS) {
  if (isVMIType(type) && !isLayoutAssignedVMIType(type)) {
    return emitInvariant(
        owner, diagOS,
        "layout-assigned VMI IR requires !pto.vmi.vreg with layout and "
        "!pto.vmi.mask with b8/b16/b32 granularity plus layout");
  }

  return success();
}

LogicalResult verifyLayoutAssignedTypeTree(Operation *owner, Type type,
                                           llvm::raw_ostream *diagOS) {
  if (failed(verifyLayoutAssignedType(owner, type, diagOS))) {
    return failure();
  }

  if (auto functionType = dyn_cast<FunctionType>(type)) {
    for (Type input : functionType.getInputs()) {
      if (failed(verifyLayoutAssignedTypeTree(owner, input, diagOS))) {
        return failure();
      }
    }
    for (Type result : functionType.getResults()) {
      if (failed(verifyLayoutAssignedTypeTree(owner, result, diagOS))) {
        return failure();
      }
    }
  }

  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return verifyLayoutAssignedTypeTree(owner, shapedType.getElementType(),
                                        diagOS);
  }

  return success();
}

template <typename TypeVerifier>
LogicalResult verifyAttributeTypes(Operation *owner, Attribute attr,
                                   llvm::raw_ostream *diagOS,
                                   TypeVerifier verifyType) {
  if (!attr) {
    return success();
  }

  if (auto typeAttr = dyn_cast<TypeAttr>(attr)) {
    if (failed(verifyType(owner, typeAttr.getValue(), diagOS))) {
      return failure();
    }
  }

  if (auto typedAttr = dyn_cast<TypedAttr>(attr)) {
    if (failed(verifyType(owner, typedAttr.getType(), diagOS))) {
      return failure();
    }
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    for (Attribute element : arrayAttr) {
      if (failed(verifyAttributeTypes(owner, element, diagOS, verifyType))) {
        return failure();
      }
    }
  }

  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
    for (NamedAttribute namedAttr : dictAttr) {
      if (failed(verifyAttributeTypes(owner, namedAttr.getValue(), diagOS,
                                      verifyType))) {
        return failure();
      }
    }
  }

  return success();
}

bool isFunctionTypeAttr(Operation *op, NamedAttribute attr) {
  return isa<func::FuncOp>(op) && attr.getName() == "function_type";
}

LogicalResult verifyNoHiddenVMIAttributeType(Operation *op, NamedAttribute attr,
                                             llvm::raw_ostream *diagOS) {
  if (isFunctionTypeAttr(op, attr)) {
    return success();
  }
  if (containsVMIType(attr.getValue())) {
    return emitInvariant(op, diagOS,
                         "VMI type appears in a non-signature attribute");
  }
  return success();
}

LogicalResult verifyOperationTypes(Operation *op, llvm::raw_ostream *diagOS) {
  if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
    FunctionType functionType = funcOp.getFunctionType();
    for (Type type : functionType.getInputs()) {
      if (failed(verifyBoundaryTypeTree(op, type, diagOS))) {
        return failure();
      }
    }
    for (Type type : functionType.getResults()) {
      if (failed(verifyBoundaryTypeTree(op, type, diagOS))) {
        return failure();
      }
    }
  }

  for (Type type : op->getOperandTypes()) {
    if (failed(verifyBoundaryTypeTree(op, type, diagOS))) {
      return failure();
    }
  }
  for (Type type : op->getResultTypes()) {
    if (failed(verifyBoundaryTypeTree(op, type, diagOS))) {
      return failure();
    }
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Type type : block.getArgumentTypes()) {
        if (failed(verifyBoundaryTypeTree(op, type, diagOS))) {
          return failure();
        }
      }
    }
  }
  for (NamedAttribute attr : op->getAttrs()) {
    if (failed(verifyNoHiddenVMIAttributeType(op, attr, diagOS))) {
      return failure();
    }
    if (failed(verifyAttributeTypes(op, attr.getValue(), diagOS,
                                    verifyBoundaryTypeTree))) {
      return failure();
    }
  }
  return success();
}

LogicalResult verifyLayoutAssignedOperationTypes(Operation *op,
                                                 llvm::raw_ostream *diagOS) {
  if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
    FunctionType functionType = funcOp.getFunctionType();
    for (Type type : functionType.getInputs()) {
      if (failed(verifyLayoutAssignedTypeTree(op, type, diagOS))) {
        return failure();
      }
    }
    for (Type type : functionType.getResults()) {
      if (failed(verifyLayoutAssignedTypeTree(op, type, diagOS))) {
        return failure();
      }
    }
  }

  for (Type type : op->getOperandTypes()) {
    if (failed(verifyLayoutAssignedTypeTree(op, type, diagOS))) {
      return failure();
    }
  }
  for (Type type : op->getResultTypes()) {
    if (failed(verifyLayoutAssignedTypeTree(op, type, diagOS))) {
      return failure();
    }
  }
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Type type : block.getArgumentTypes()) {
        if (failed(verifyLayoutAssignedTypeTree(op, type, diagOS))) {
          return failure();
        }
      }
    }
  }
  for (NamedAttribute attr : op->getAttrs()) {
    if (failed(verifyNoHiddenVMIAttributeType(op, attr, diagOS))) {
      return failure();
    }
    if (failed(verifyAttributeTypes(op, attr.getValue(), diagOS,
                                    verifyLayoutAssignedTypeTree))) {
      return failure();
    }
  }
  return success();
}

LogicalResult verifyLayoutHelperSupport(Operation *op,
                                        llvm::raw_ostream *diagOS);

LogicalResult verifyLayoutSemanticSupport(Operation *op,
                                          llvm::raw_ostream *diagOS);

LogicalResult verifyOperationBoundary(Operation *op,
                                      llvm::raw_ostream *diagOS) {
  if (failed(verifyOperationTypes(op, diagOS))) {
    return failure();
  }

  if (!hasVMIType(op)) {
    return success();
  }

  if (isVMIHelperOp(op)) {
    return emitInvariant(
        op, diagOS,
        "VMI helper op appears before layout assignment or VMI-to-VPTO");
  }

  if (isVMISemanticOp(op) || isStructuralOp(op)) {
    return success();
  }

  return emitInvariant(op, diagOS,
                       "VMI typed value is used by a non-VMI semantic op");
}

LogicalResult verifyLayoutAssignedOperation(Operation *op,
                                            llvm::raw_ostream *diagOS,
                                            bool verifyHelperSupports = true) {
  if (failed(verifyLayoutAssignedOperationTypes(op, diagOS))) {
    return failure();
  }

  if (!hasVMIType(op)) {
    return success();
  }

  if (isVMIHelperOp(op)) {
    if (isVMILayoutHelperOp(op)) {
      return verifyHelperSupports ? verifyLayoutHelperSupport(op, diagOS)
                                  : success();
    }
    return emitInvariant(
        op, diagOS,
        "VMI pack/unpack helper appears before VMI-to-VPTO physicalization");
  }

  if (isVMISemanticOp(op)) {
    return verifyLayoutSemanticSupport(op, diagOS);
  }
  if (isStructuralOp(op)) {
    return success();
  }

  return emitInvariant(op, diagOS,
                       "VMI typed value is used by a non-VMI semantic op");
}

LogicalResult verifyLayoutHelperSupport(Operation *op,
                                        llvm::raw_ostream *diagOS) {
  VMILayoutSupport supports;

  if (auto ensure = dyn_cast<VMIEnsureLayoutOp>(op)) {
    auto sourceType = cast<VMIVRegType>(ensure.getSource().getType());
    auto resultType = cast<VMIVRegType>(ensure.getResult().getType());
    std::string reason;
    if (failed(supports.getEnsureLayoutFact(sourceType, resultType, &reason))) {
      return emitHelperMaterializationContract(
          op, sourceType, resultType, "pto.vmi.ensure_layout", reason, diagOS);
    }
    return success();
  }

  if (auto ensure = dyn_cast<VMIEnsureMaskLayoutOp>(op)) {
    auto sourceType = cast<VMIMaskType>(ensure.getSource().getType());
    auto resultType = cast<VMIMaskType>(ensure.getResult().getType());
    std::string reason;
    if (failed(
            supports.getEnsureMaskLayoutFact(sourceType, resultType, &reason))) {
      return emitHelperMaterializationContract(op, sourceType, resultType,
                                               "pto.vmi.ensure_mask_layout",
                                               reason, diagOS);
    }
    return success();
  }

  return success();
}

LogicalResult verifyLayoutSemanticSupport(Operation *op,
                                          llvm::raw_ostream *diagOS) {
  VMILayoutSupport supports;

  if (auto store = dyn_cast<VMIStoreOp>(op)) {
    auto valueType = cast<VMIVRegType>(store.getValue().getType());
    VMILayoutAttr layout = valueType.getLayoutAttr();
    if (!layout || layout.isContiguous()) {
      return success();
    }

    std::string reason;
    if (failed(supports.getStoreLayoutFact(valueType, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS,
          "pto.vmi.store has no registered contiguous-memory layout support",
          reason);
    }
    return success();
  }

  if (auto load = dyn_cast<VMIGroupLoadOp>(op)) {
    auto resultType = cast<VMIVRegType>(load.getResult().getType());
    VMILayoutAttr layout = resultType.getLayoutAttr();
    if (!layout) {
      return success();
    }

    std::string reason;
    if (failed(supports.getGroupLoadLayoutFact(load, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS,
          "pto.vmi.group_load has no registered layout support", reason);
    }
    return success();
  }

  if (auto load = dyn_cast<VMIGroupSlotLoadOp>(op)) {
    auto resultType = cast<VMIVRegType>(load.getResult().getType());
    std::string reason;
    if (failed(supports.getGroupSlotLoadLayoutFact(
            resultType, load.getNumGroupsAttr().getInt(), &reason))) {
      return emitLayoutSupportContract(
          op, diagOS,
          "pto.vmi.group_slot_load has no registered layout support", reason);
    }
    return success();
  }

  if (auto load = dyn_cast<VMIGroupBroadcastLoadOp>(op)) {
    std::string reason;
    if (failed(supports.getGroupBroadcastLoadSupport(load, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS,
          "pto.vmi.group_broadcast_load has no registered layout support",
          reason);
    }
    return success();
  }

  if (auto store = dyn_cast<VMIGroupStoreOp>(op)) {
    auto valueType = cast<VMIVRegType>(store.getValue().getType());
    VMILayoutAttr layout = valueType.getLayoutAttr();
    if (!layout || !layout.isGroupSlots()) {
      return success();
    }

    std::string reason;
    if (failed(supports.getGroupStoreLayoutFact(
            valueType, store.getNumGroupsAttr().getInt(), &reason))) {
      return emitLayoutSupportContract(
          op, diagOS,
          "pto.vmi.group_store has no registered group_slots layout support",
          reason);
    }
    return success();
  }

  if (auto reduce = dyn_cast<VMIGroupReduceAddFOp>(op)) {
    auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
    VMILayoutAttr layout = resultType.getLayoutAttr();
    if (!layout || !layout.isGroupSlots()) {
      return success();
    }

    std::string reason;
    if (failed(supports.getGroupReduceAddFSupport(reduce, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS,
          "pto.vmi.group_reduce_addf has no registered group_slots layout "
          "support",
          reason);
    }
    return success();
  }

  if (auto reduce = dyn_cast<VMIGroupReduceMaxFOp>(op)) {
    auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
    VMILayoutAttr layout = resultType.getLayoutAttr();
    if (!layout || !layout.isGroupSlots()) {
      return success();
    }

    std::string reason;
    if (failed(supports.getGroupReduceMaxFSupport(reduce, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS,
          "pto.vmi.group_reduce_maxf has no registered group_slots layout "
          "support",
          reason);
    }
    return success();
  }

  if (auto reduce = dyn_cast<VMIGroupReduceMinFOp>(op)) {
    auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
    VMILayoutAttr layout = resultType.getLayoutAttr();
    if (!layout || !layout.isGroupSlots()) {
      return success();
    }

    std::string reason;
    if (failed(supports.getGroupReduceMinFSupport(reduce, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS,
          "pto.vmi.group_reduce_minf has no registered group_slots layout "
          "support",
          reason);
    }
    return success();
  }

  if (auto reduce = dyn_cast<VMIGroupReduceAddIOp>(op)) {
    auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
    VMILayoutAttr layout = resultType.getLayoutAttr();
    if (!layout || !layout.isGroupSlots()) {
      return success();
    }

    std::string reason;
    if (failed(supports.getGroupReduceAddISupport(reduce, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS,
          "pto.vmi.group_reduce_addi has no registered group_slots layout "
          "support",
          reason);
    }
    return success();
  }

  if (auto reduce = dyn_cast<VMIGroupReduceMaxIOp>(op)) {
    auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
    VMILayoutAttr layout = resultType.getLayoutAttr();
    if (!layout || !layout.isGroupSlots()) {
      return success();
    }

    std::string reason;
    if (failed(supports.getGroupReduceMaxISupport(reduce, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS,
          "pto.vmi.group_reduce_maxi has no registered group_slots layout "
          "support",
          reason);
    }
    return success();
  }

  if (auto reduce = dyn_cast<VMIGroupReduceMinIOp>(op)) {
    auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
    VMILayoutAttr layout = resultType.getLayoutAttr();
    if (!layout || !layout.isGroupSlots()) {
      return success();
    }

    std::string reason;
    if (failed(supports.getGroupReduceMinISupport(reduce, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS,
          "pto.vmi.group_reduce_mini has no registered group_slots layout "
          "support",
          reason);
    }
    return success();
  }

  if (auto broadcast = dyn_cast<VMIGroupBroadcastOp>(op)) {
    auto sourceType = cast<VMIVRegType>(broadcast.getSource().getType());
    VMILayoutAttr layout = sourceType.getLayoutAttr();
    if (!layout || !layout.isGroupSlots() || layout.getSlots() <= 0) {
      return success();
    }

    std::string reason;
    if (failed(supports.getGroupBroadcastSupport(broadcast, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS,
          "pto.vmi.group_broadcast has no registered layout support", reason);
    }
    return success();
  }

  if (auto hist = dyn_cast<VMIVdhistOp>(op)) {
    std::string reason;
    if (failed(supports.getVdhistSupport(hist, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS, "pto.vmi.vdhist has no registered histogram support",
          reason);
    }
    return success();
  }

  if (auto hist = dyn_cast<VMIVchistOp>(op)) {
    std::string reason;
    if (failed(supports.getVchistSupport(hist, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS, "pto.vmi.vchist has no registered histogram support",
          reason);
    }
    return success();
  }

  if (auto truncf = dyn_cast<VMITruncFOp>(op)) {
    std::string reason;
    if (failed(supports.getTruncFSupport(truncf, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS, "pto.vmi.truncf has no registered layout support",
          reason);
    }
    return success();
  }

  if (auto extf = dyn_cast<VMIExtFOp>(op)) {
    std::string reason;
    if (failed(supports.getExtFSupport(extf, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS, "pto.vmi.extf has no registered layout support", reason);
    }
    return success();
  }

  if (auto bitcast = dyn_cast<VMIBitcastOp>(op)) {
    std::string reason;
    if (failed(supports.getBitcastSupport(bitcast, &reason))) {
      return emitLayoutSupportContract(
          op, diagOS, "pto.vmi.bitcast has no registered layout support",
          reason);
    }
    return success();
  }

  return success();
}

struct PTOValidateVMIIRPass
    : public mlir::pto::impl::PTOValidateVMIIRBase<PTOValidateVMIIRPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOValidateVMIIRPass)

  void runOnOperation() override {
    if (failed(validateVMIProducerBoundaryIR(getOperation(), &llvm::errs()))) {
      signalPassFailure();
    }
  }
};

struct PTOValidateVMILayoutIRPass
    : public mlir::pto::impl::PTOValidateVMILayoutIRBase<
          PTOValidateVMILayoutIRPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOValidateVMILayoutIRPass)

  void runOnOperation() override {
    if (failed(validateVMILayoutAssignedIR(getOperation(), &llvm::errs()))) {
      signalPassFailure();
    }
  }
};

} // namespace

LogicalResult
mlir::pto::validateVMIProducerBoundaryIR(ModuleOp module,
                                         llvm::raw_ostream *diagOS) {
  WalkResult result = module.walk([&](Operation *op) {
    if (failed(verifyOperationBoundary(op, diagOS))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult mlir::pto::validateVMILayoutAssignedIR(
    ModuleOp module, llvm::raw_ostream *diagOS, bool verifyHelperSupports) {
  WalkResult result = module.walk([&](Operation *op) {
    if (failed(verifyLayoutAssignedOperation(op, diagOS, verifyHelperSupports))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

std::unique_ptr<Pass> mlir::pto::createPTOValidateVMIIRPass() {
  return std::make_unique<PTOValidateVMIIRPass>();
}

std::unique_ptr<Pass> mlir::pto::createPTOValidateVMILayoutIRPass() {
  return std::make_unique<PTOValidateVMILayoutIRPass>();
}

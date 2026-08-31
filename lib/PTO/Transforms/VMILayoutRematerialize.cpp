// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILayoutRematerialize.cpp - Rematerialize VMI producers ----------===//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMILayoutSupport.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMILAYOUTREMATERIALIZE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool hasConcreteLayout(VMIVRegType type) {
  return type && static_cast<bool>(type.getLayoutAttr());
}

static bool hasConcreteLayout(VMIMaskType type) {
  return type && static_cast<bool>(type.getLayoutAttr());
}

static Value materializeDataLayout(Value value, VMIVRegType resultType,
                                   Location loc, OpBuilder &builder) {
  auto sourceType = dyn_cast<VMIVRegType>(value.getType());
  if (!sourceType || sourceType == resultType) {
    return value;
  }

  return builder.create<VMIEnsureLayoutOp>(loc, resultType, value).getResult();
}

template <typename ExtOp>
static std::optional<Value>
rematerializeWidenExt(ExtOp op, VMIVRegType resultType, Location loc,
                      OpBuilder &builder) {
  auto sourceType = dyn_cast<VMIVRegType>(op.getSource().getType());
  if (!sourceType || !hasConcreteLayout(resultType)) {
    return std::nullopt;
  }

  VMILayoutSupport supports;
  FailureOr<VMILayoutAttr> sourceLayout =
      supports.getWidenSourceLayoutForResultLayout(sourceType, resultType,
                                                   resultType.getLayoutAttr());
  if (failed(sourceLayout)) {
    return std::nullopt;
  }

  auto rematSourceType =
      VMIVRegType::get(sourceType.getContext(), sourceType.getElementCount(),
                       sourceType.getElementType(), *sourceLayout);
  if (sourceType != rematSourceType) {
    if (failed(supports.getEnsureLayoutFact(sourceType, rematSourceType))) {
      return std::nullopt;
    }

    FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
    FailureOr<int64_t> rematSourceArity =
        getVMIPhysicalArity(rematSourceType);
    if (failed(sourceArity) || failed(rematSourceArity) ||
        *rematSourceArity > *sourceArity) {
      return std::nullopt;
    }
  }
  Value rematSource =
      materializeDataLayout(op.getSource(), rematSourceType, loc, builder);
  return builder.create<ExtOp>(loc, resultType, rematSource).getResult();
}

static bool isRematerializableElementwiseOp(Operation *op) {
  return isa<VMIVaddOp, VMIVsubOp, VMIVmulOp, VMIVdivOp, VMIVminOp, VMIVmaxOp,
             VMIVandOp, VMIVorOp, VMIVxorOp, VMIVshlOp, VMIVshrOp, VMIVnegOp,
             VMIVabsOp, VMIVsqrtOp, VMIVexpOp, VMIVlnOp, VMIVreluOp, VMIVnotOp,
             VMIVmulaOp>(op);
}

static std::optional<Value> rematerializeElementwiseOp(Operation *op,
                                                       VMIVRegType resultType,
                                                       Location loc,
                                                       OpBuilder &builder) {
  if (!isRematerializableElementwiseOp(op)) {
    return std::nullopt;
  }

  SmallVector<Value> operands;
  operands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    if (auto dataType = dyn_cast<VMIVRegType>(operand.getType())) {
      auto targetType = VMIVRegType::get(
          dataType.getContext(), dataType.getElementCount(),
          dataType.getElementType(), resultType.getLayoutAttr());
      operands.push_back(
          materializeDataLayout(operand, targetType, loc, builder));
      continue;
    }
    if (auto maskType = dyn_cast<VMIMaskType>(operand.getType())) {
      auto targetType = VMIMaskType::get(
          maskType.getContext(), maskType.getElementCount(),
          maskType.getGranularity(), resultType.getLayoutAttr());
      if (maskType != targetType) {
        operand =
            builder.create<VMIEnsureMaskLayoutOp>(loc, targetType, operand)
                .getResult();
      }
    }
    operands.push_back(operand);
  }

  OperationState state(loc, op->getName());
  state.addOperands(operands);
  state.addTypes(resultType);
  state.addAttributes(op->getAttrs());
  return builder.create(state)->getResult(0);
}

static std::optional<Value> rematerializeDataProducer(Value value,
                                                      VMIVRegType resultType,
                                                      Location loc,
                                                      OpBuilder &builder) {
  if (!hasConcreteLayout(resultType)) {
    return std::nullopt;
  }

  if (auto extf = value.getDefiningOp<VMIExtFOp>()) {
    return rematerializeWidenExt(extf, resultType, loc, builder);
  }
  if (auto extsi = value.getDefiningOp<VMIExtSIOp>()) {
    return rematerializeWidenExt(extsi, resultType, loc, builder);
  }
  if (auto extui = value.getDefiningOp<VMIExtUIOp>()) {
    return rematerializeWidenExt(extui, resultType, loc, builder);
  }

  if (Operation *op = value.getDefiningOp()) {
    if (auto result =
            rematerializeElementwiseOp(op, resultType, loc, builder)) {
      return result;
    }
  }

  if (auto constant = value.getDefiningOp<VMIConstantOp>()) {
    auto denseAttr = dyn_cast<DenseElementsAttr>(constant.getValue());
    if (denseAttr && denseAttr.isSplat()) {
      return builder.create<VMIConstantOp>(loc, resultType, constant.getValue())
          .getResult();
    }
  }

  if (auto broadcast = value.getDefiningOp<VMIBroadcastOp>()) {
    return builder.create<VMIBroadcastOp>(loc, resultType, broadcast.getValue())
        .getResult();
  }

  if (auto iota = value.getDefiningOp<VMIIotaOp>()) {
    return builder
        .create<VMIIotaOp>(loc, resultType, iota.getBase(), iota.getOrderAttr())
        .getResult();
  }

  if (auto groupIota = value.getDefiningOp<VMIGroupIotaOp>()) {
    VMILayoutAttr resultLayout = resultType.getLayoutAttr();
    if (!resultLayout || !resultLayout.isContiguous()) {
      return std::nullopt;
    }
    return builder
        .create<VMIGroupIotaOp>(loc, resultType, groupIota.getBase(),
                                groupIota.getOrderAttr(),
                                groupIota.getGroupAttr())
        .getResult();
  }

  return std::nullopt;
}

static std::optional<Value> rematerializeMaskProducer(Value value,
                                                      VMIMaskType resultType,
                                                      Location loc,
                                                      OpBuilder &builder) {
  if (!hasConcreteLayout(resultType)) {
    return std::nullopt;
  }

  if (auto createMask = value.getDefiningOp<VMICreateMaskOp>()) {
    return builder
        .create<VMICreateMaskOp>(loc, resultType, createMask.getActiveLanes())
        .getResult();
  }

  if (auto createGroupMask = value.getDefiningOp<VMICreateGroupMaskOp>()) {
    return builder
        .create<VMICreateGroupMaskOp>(loc, resultType,
                                      createGroupMask.getActiveElemsPerGroup(),
                                      createGroupMask.getNumGroupsAttr(),
                                      createGroupMask.getGroupSizeAttr())
        .getResult();
  }

  if (auto constantMask = value.getDefiningOp<VMIConstantMaskOp>()) {
    return builder
        .create<VMIConstantMaskOp>(loc, resultType, constantMask.getValueAttr())
        .getResult();
  }

  return std::nullopt;
}

static bool tryReplaceDataEnsure(VMIEnsureLayoutOp ensure) {
  auto resultType = dyn_cast<VMIVRegType>(ensure.getResult().getType());
  if (!resultType) {
    return false;
  }

  OpBuilder builder(ensure);
  auto result = rematerializeDataProducer(ensure.getSource(), resultType,
                                          ensure->getLoc(), builder);
  if (!result) {
    return false;
  }

  ensure.getResult().replaceAllUsesWith(*result);
  ensure.erase();
  return true;
}

static bool tryRematerializeTruncIThroughSourceEnsure(VMITruncIOp trunc) {
  auto resultType = dyn_cast<VMIVRegType>(trunc.getResult().getType());
  if (!resultType || !hasConcreteLayout(resultType)) {
    return false;
  }

  auto ensure = trunc.getSource().getDefiningOp<VMIEnsureLayoutOp>();
  if (!ensure) {
    return false;
  }

  auto originalSourceType = dyn_cast<VMIVRegType>(ensure.getSource().getType());
  if (!originalSourceType || !hasConcreteLayout(originalSourceType)) {
    return false;
  }
  VMILayoutAttr originalSourceLayout = originalSourceType.getLayoutAttr();
  if (!originalSourceLayout.isDeinterleaved()) {
    return false;
  }

  VMILayoutSupport supports;
  FailureOr<VMICastLayoutFact> fact = supports.getCastLayoutFactForSourceLayout(
      originalSourceType, resultType, originalSourceLayout);
  if (failed(fact)) {
    return false;
  }

  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (resultBits == mlir::pto::kValue8 &&
      !cast<IntegerType>(resultType.getElementType()).isUnsigned()) {
    return false;
  }

  VMILayoutAttr rematResultLayout = fact->resultLayout;
  auto rematResultType =
      VMIVRegType::get(resultType.getContext(), resultType.getElementCount(),
                       resultType.getElementType(), rematResultLayout);
  if (rematResultType == resultType) {
    return false;
  }

  OpBuilder builder(trunc);
  Value remat = builder
                    .create<VMITruncIOp>(trunc->getLoc(), rematResultType,
                                         ensure.getSource(),
                                         trunc.getSaturateAttr())
                    .getResult();
  Value replacement =
      materializeDataLayout(remat, resultType, trunc->getLoc(), builder);
  trunc.getResult().replaceAllUsesWith(replacement);
  trunc.erase();
  return true;
}

template <typename EnsureOp> static bool tryReplaceMaskEnsure(EnsureOp ensure) {
  auto resultType = dyn_cast<VMIMaskType>(ensure.getResult().getType());
  if (!resultType) {
    return false;
  }

  OpBuilder builder(ensure);
  auto result = rematerializeMaskProducer(ensure.getSource(), resultType,
                                          ensure->getLoc(), builder);
  if (!result) {
    return false;
  }

  ensure.getResult().replaceAllUsesWith(*result);
  ensure.erase();
  return true;
}

struct VMILayoutRematerializePass
    : public mlir::pto::impl::VMILayoutRematerializeBase<
          VMILayoutRematerializePass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMILayoutRematerializePass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> helpers;
      module.walk([&](Operation *op) {
        if (isa<VMIEnsureLayoutOp, VMIEnsureMaskLayoutOp,
                VMIEnsureMaskGranularityOp, VMITruncIOp>(op)) {
          helpers.push_back(op);
        }
      });

      for (Operation *op : helpers) {
        if (op->getBlock() == nullptr) {
          continue;
        }

        if (auto ensure = dyn_cast<VMIEnsureLayoutOp>(op)) {
          changed |= tryReplaceDataEnsure(ensure);
          continue;
        }

        if (auto ensure = dyn_cast<VMIEnsureMaskLayoutOp>(op)) {
          changed |= tryReplaceMaskEnsure(ensure);
          continue;
        }

        if (auto ensure = dyn_cast<VMIEnsureMaskGranularityOp>(op)) {
          changed |= tryReplaceMaskEnsure(ensure);
          continue;
        }

        if (auto trunc = dyn_cast<VMITruncIOp>(op)) {
          changed |= tryRematerializeTruncIThroughSourceEnsure(trunc);
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMILayoutRematerializePass() {
  return std::make_unique<VMILayoutRematerializePass>();
}

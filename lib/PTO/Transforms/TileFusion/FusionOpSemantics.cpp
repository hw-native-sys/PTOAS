// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/TileFusion/FusionOpSemantics.h"

#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir {
namespace pto {

static constexpr llvm::StringLiteral kVmiFusionBoundaryAttr =
    "pto.vmi.fusion.boundary";

static FusionComputeFamily getFusionComputeFamily(StringRef opName) {
  return llvm::StringSwitch<FusionComputeFamily>(opName)
      .Cases("tadd", "tsub", "tmul", "tdiv", "tmax", "tmin",
             FusionComputeFamily::Elementwise)
      .Cases("tadds", "tsubs", "tmuls", "tdivs", "tmaxs", "tmins",
             FusionComputeFamily::Elementwise)
      .Cases("texp", "tabs", "tneg", "trecip", "tsqrt", "trsqrt",
             FusionComputeFamily::Elementwise)
      .Case("tmov", FusionComputeFamily::Elementwise)
      .Case("texpands", FusionComputeFamily::ScalarExpand)
      .Cases("trowexpandsub", "trowexpandmul", "trowexpanddiv",
             FusionComputeFamily::RowBroadcastBinary)
      .Cases("tcolexpandsub", "tcolexpandadd", "tcolexpandmul",
             "tcolexpanddiv",
             FusionComputeFamily::ColBroadcastBinary)
      .Cases("trowsum", "trowmax", "trowmin", FusionComputeFamily::ReduceRow)
      .Cases("tcolsum", "tcolmax", "tcolmin", FusionComputeFamily::ReduceCol)
      .Case("tcvt", FusionComputeFamily::Convert)
      .Default(FusionComputeFamily::Unknown);
}

bool isSupportedPreFusionComputeOp(StringRef opName) {
  return getFusionComputeFamily(opName) != FusionComputeFamily::Unknown;
}

bool isFusionTransparentScaffold(Operation *op) {
  if (!op || op->hasTrait<OpTrait::IsTerminator>() ||
      !op->getRegions().empty() || isa<CallOpInterface>(op))
    return false;

  // These PTO operations only construct logical storage/view descriptors.
  // Keeping them in a loose region is required to preserve SSA dominance
  // when a later compute consumes the descriptor they define.
  if (isa<pto::AllocTileOp, pto::SubViewOp, pto::MakeTensorViewOp,
          pto::PartitionViewOp>(op))
    return true;

  // Scalar/index plumbing from other dialects is transparent when it is
  // effect-free. Unknown PTO operations remain conservative boundaries: a
  // new tile/data-movement op must be classified explicitly above or as a
  // supported compute family.
  if (op->getDialect() == op->getContext()->getLoadedDialect<pto::PTODialect>())
    return false;
  if (!isMemoryEffectFree(op))
    return false;

  // Keep this generic allowance narrow: it exists for arith/index plumbing,
  // not for moving arbitrary tensor/memref transformations into a region.
  return llvm::all_of(op->getOperandTypes(),
                      [](Type type) { return type.isIntOrIndexOrFloat(); }) &&
         llvm::all_of(op->getResultTypes(),
                      [](Type type) { return type.isIntOrIndexOrFloat(); });
}

static bool isTileFusionTileValue(Value value) {
  return isa<pto::TileBufType>(value.getType());
}

static SmallVector<Value, 2> collectNormalizedTileOutputs(Operation *op) {
  SmallVector<Value, 2> outputs;

  if (auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(op)) {
    for (Value init : dpsIface.getDpsInits()) {
      if (isTileFusionTileValue(init)) {
        outputs.push_back(init);
      }
    }
    if (!outputs.empty())
      return outputs;
  }

  for (Value result : op->getResults()) {
    if (isTileFusionTileValue(result))
      outputs.push_back(result);
  }
  return outputs;
}

static StringRef getTileFusionOpName(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  opName.consume_front("pto.");
  return opName;
}

FailureOr<FusionOpSemantics> getFusionOpSemantics(Operation *op) {
  FusionOpSemantics semantics;
  semantics.op = op;
  semantics.opName = getTileFusionOpName(op).str();

  if (auto reshape = dyn_cast<pto::TReshapeOp>(op)) {
    semantics.kind = FusionOpKind::LocalBoundary;
    semantics.opName = "treshape";
    semantics.tileInputs.push_back(reshape.getSrc());
    semantics.tileOutputs.push_back(reshape.getResult());
    return semantics;
  }

  // Candidate selection runs before the VMI fusion planner. A non-VMI
  // fallback remains a valid TileLib implementation, but it is not a VMI
  // compute node: local fallbacks split a VMI loop-fusion run while remaining
  // inside the loose FusionRegion; hard fallbacks split the enclosing region.
  // The ordinary (non-VMI) fusion pipeline selects its candidates after
  // planning, so this does not change legacy MI behavior.
  if (auto boundary = op->getAttrOfType<StringAttr>(kVmiFusionBoundaryAttr)) {
    semantics.kind = boundary.getValue() == "local"
                         ? FusionOpKind::LocalBoundary
                         : FusionOpKind::HardBoundary;
    semantics.tileOutputs = collectNormalizedTileOutputs(op);

    SmallVector<unsigned, 4> dpsInitOperandNumbers;
    if (auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(op)) {
      for (OpOperand &dpsInit : dpsIface.getDpsInitsMutable())
        dpsInitOperandNumbers.push_back(dpsInit.getOperandNumber());
    }
    for (OpOperand &operand : op->getOpOperands()) {
      if (llvm::is_contained(dpsInitOperandNumbers,
                             operand.getOperandNumber()))
        continue;
      Value value = operand.get();
      if (isTileFusionTileValue(value))
        semantics.tileInputs.push_back(value);
      else
        semantics.scalarInputs.push_back(value);
    }
    return semantics;
  }

  // Whitelist-based compute classification. Any op NOT in
  // getFusionComputeFamily's whitelist is a HardBoundary, which means it is
  // excluded from computeNodes by FusionAnalysis and, in the
  // VMIUBDisjointStrategyEngine, appears as the "preceded-by-non-plannable"
  // F3 boundary that closes the current fusion group. This is what keeps
  // sync/DMA/unknown ops (wait_flag, mem_bar, tload/tstore, ...) from
  // being merged across — they fall through here because they are Unknown, not
  // because they are listed.
  //
  // CONTRACT: adding a new plannable TileOp compute op REQUIRES registering it
  // in getFusionComputeFamily above. Forgetting to do so silently turns it
  // into a HardBoundary, and VMIUBDisjointStrategyEngine will then wrongly
  // split the two adjacent groups it was supposed to join. This is the inverse
  // of the deleted PTOPlanVmiFusionRegion pass, which used a sync-op blacklist:
  // there, adding a compute op was safe by default; here, it is unsafe by
  // default. See test/lit/vpto/vmi_plan_f3_boundary.pto.
  semantics.computeFamily = getFusionComputeFamily(semantics.opName);
  if (semantics.computeFamily == FusionComputeFamily::Unknown) {
    semantics.kind = FusionOpKind::HardBoundary;
    return semantics;
  }

  auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(op);
  if (!dpsIface && op->getNumResults() == 0) {
    semantics.kind = FusionOpKind::HardBoundary;
    return semantics;
  }

  semantics.kind = FusionOpKind::Compute;
  semantics.tileOutputs = collectNormalizedTileOutputs(op);
  // EmitC still carries tile operations on memrefs. They are not eligible for
  // tile-native fusion, but remain valid hard boundaries in the shared DFG.
  if (semantics.tileOutputs.empty()) {
    semantics.kind = FusionOpKind::HardBoundary;
    return semantics;
  }

  SmallVector<unsigned, 4> dpsInitOperandNumbers;
  if (dpsIface) {
    for (OpOperand &dpsInit : dpsIface.getDpsInitsMutable())
      dpsInitOperandNumbers.push_back(dpsInit.getOperandNumber());
  }

  for (OpOperand &operand : op->getOpOperands()) {
    if (llvm::is_contained(dpsInitOperandNumbers, operand.getOperandNumber()))
      continue;

    Value value = operand.get();
    if (isTileFusionTileValue(value)) {
      semantics.tileInputs.push_back(value);
    } else {
      semantics.scalarInputs.push_back(value);
    }
  }

  if (semantics.tileInputs.empty()) {
    for (Value output : semantics.tileOutputs) {
      if (!isa<pto::TileBufType>(output.getType()))
        return failure();
    }
  }

  return semantics;
}

} // namespace pto
} // namespace mlir

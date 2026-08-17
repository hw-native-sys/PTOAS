// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/TileFusion/FusionOpSemantics.h"

#include "PTO/Support/CodeConstants.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir {
namespace pto {

static FusionComputeFamily getFusionComputeFamily(StringRef opName) {
  return llvm::StringSwitch<FusionComputeFamily>(opName)
      .Cases("tadd", "tsub", "tmul", "tdiv", "tmax", "tmin",
             FusionComputeFamily::Elementwise)
      .Cases("tadds", "tsubs", "tmuls", "tdivs", "tmaxs", "tmins",
             FusionComputeFamily::Elementwise)
      .Case("texp", FusionComputeFamily::Elementwise)
      .Case("texpands", FusionComputeFamily::ScalarExpand)
      .Cases("trowexpandsub", "trowexpandmul", "trowexpanddiv",
             FusionComputeFamily::RowBroadcastBinary)
      .Cases("trowsum", "trowmax", "trowmin", FusionComputeFamily::ReduceRow)
      .Cases("tcolsum", "tcolmax", "tcolmin", FusionComputeFamily::ReduceCol)
      .Default(FusionComputeFamily::Unknown);
}

bool isSupportedPreFusionComputeOp(StringRef opName) {
  return getFusionComputeFamily(opName) != FusionComputeFamily::Unknown;
}

static bool isTileFusionTileValue(Value value) {
  return isa<pto::TileBufType>(value.getType());
}

static SmallVector<Value, mlir::pto::kValue2> collectNormalizedTileOutputs(Operation *op) {
  SmallVector<Value, mlir::pto::kValue2> outputs;

  if (auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(op)) {
    for (Value init : dpsIface.getDpsInits()) {
      if (isTileFusionTileValue(init)) {
        outputs.push_back(init);
      }
    }
    if (!outputs.empty()) {
      return outputs;
    }
  }

  for (Value result : op->getResults()) {
    if (isTileFusionTileValue(result)) {
      outputs.push_back(result);
    }
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
  if (semantics.tileOutputs.empty()) {
    return failure();
  }

  SmallVector<unsigned, mlir::pto::kValue4> dpsInitOperandNumbers;
  if (dpsIface) {
    for (OpOperand &dpsInit : dpsIface.getDpsInitsMutable()) {
      dpsInitOperandNumbers.push_back(dpsInit.getOperandNumber());
    }
  }

  for (OpOperand &operand : op->getOpOperands()) {
    if (llvm::is_contained(dpsInitOperandNumbers, operand.getOperandNumber())) {
      continue;
    }

    Value value = operand.get();
    if (isTileFusionTileValue(value)) {
      semantics.tileInputs.push_back(value);
    } else {
      semantics.scalarInputs.push_back(value);
    }
  }

  if (semantics.tileInputs.empty()) {
    for (Value output : semantics.tileOutputs) {
      if (!isa<pto::TileBufType>(output.getType())) {
        return failure();
      }
    }
  }

  return semantics;
}

} // namespace pto
} // namespace mlir

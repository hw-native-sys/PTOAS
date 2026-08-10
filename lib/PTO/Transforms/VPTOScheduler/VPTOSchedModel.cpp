// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VPTOSchedModel.cpp - VPTO scheduling target model -----------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedModel.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/VPTOScheduling.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
enum ResourceID : unsigned {
  ScalarResource,
  VectorResource,
  MTEResource,
  CubeResource,
  ControlResource,
  UnknownResource,
};

enum PressureSetID : unsigned {
  VectorPressure,
  PredicatePressure,
  ScalarPressure,
  AddressPressure,
  AlignPressure,
  SpecialPressure,
};

enum SchedClassID : unsigned {
  StructuralClass,
  ScalarClass,
  VectorClass,
  MTEClass,
  CubeClass,
  UnknownClass,
};
} // namespace

VPTOGenericA5SchedModel::VPTOGenericA5SchedModel() {
  machine.target = "a5";
  machine.version = "generic-a5-v1";
  machine.issueWidth = 1;
  machine.microOpBufferSize = 0;
  machine.completeness = VPTOSchedModelCompleteness::Minimal;

  resources = {
      {ScalarResource, "scalar", 1, 0, {}},
      {VectorResource, "vector", 1, 0, {}},
      {MTEResource, "mte", 1, 0, {}},
      {CubeResource, "cube", 1, 0, {}},
      {ControlResource, "control", 1, 0, {}},
      {UnknownResource, "unknown", 1, 0, {}},
  };
  pressureSets = {
      {VectorPressure, "vector", std::nullopt, 1.0, 1.0},
      {PredicatePressure, "predicate", std::nullopt, 1.0, 1.0},
      {ScalarPressure, "scalar", std::nullopt, 1.0, 1.0},
      {AddressPressure, "address", std::nullopt, 1.0, 1.0},
      {AlignPressure, "align", std::nullopt, 1.0, 1.0},
      {SpecialPressure, "special", std::nullopt, 1.0, 1.0},
  };
  schedClasses = {
      {StructuralClass, "structural", true, 0, 0, {}, {}},
      {ScalarClass, "scalar", true, 1, 1, {{ScalarResource, 0, 1, 1}}, {}},
      {VectorClass, "vector", true, 1, 1, {{VectorResource, 0, 1, 1}}, {}},
      {MTEClass, "mte", true, 1, 2, {{MTEResource, 0, 1, 1}}, {}},
      {CubeClass, "cube", true, 1, 4, {{CubeResource, 0, 1, 1}}, {}},
      {UnknownClass, "unknown", false, 1, 1, {{UnknownResource, 0, 1, 1}}, {}},
  };
}

const VPTOSchedClass &
VPTOGenericA5SchedModel::getSchedClass(Operation *op) const {
  if (!op || classifyVPTOSchedulingOp(op) == VPTOSchedulingClass::Structural)
    return schedClasses[StructuralClass];
  if (isa<VectorMicroOpInterface>(op))
    return schedClasses[VectorClass];
  if (isa<MteOpInterface>(op))
    return schedClasses[MTEClass];
  if (isa<CubeMicroOpInterface>(op))
    return schedClasses[CubeClass];
  if (isa<SimtOpInterface>(op))
    return schedClasses[ScalarClass];
  return schedClasses[UnknownClass];
}

SmallVector<VPTORegPressureContribution>
VPTOGenericA5SchedModel::getPressure(Value value) const {
  if (!value)
    return {};
  Type type = value.getType();
  if (isa<VRegType>(type))
    return {{VectorPressure, 1}};
  if (isa<MaskType>(type))
    return {{PredicatePressure, 1}};
  if (isa<AlignType>(type))
    return {{AlignPressure, 1}};
  if (isa<PtrType, BaseMemRefType>(type))
    return {{AddressPressure, 1}};
  if (type.isIntOrIndexOrFloat())
    return {{ScalarPressure, 1}};
  return {{SpecialPressure, 1}};
}

StringRef mlir::pto::stringifyVPTOSchedModelCompleteness(
    VPTOSchedModelCompleteness completeness) {
  switch (completeness) {
  case VPTOSchedModelCompleteness::Minimal:
    return "minimal";
  case VPTOSchedModelCompleteness::Partial:
    return "partial";
  case VPTOSchedModelCompleteness::Complete:
    return "complete";
  }
  llvm_unreachable("unknown VPTO scheduling model completeness");
}

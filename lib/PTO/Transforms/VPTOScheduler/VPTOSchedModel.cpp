// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedModel.cpp - VPTO scheduling target model -----------------===//

#include "PTO/Transforms/VPTOScheduler/VPTOSchedModel.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/VPTOScheduling.h"

#include "mlir/IR/BuiltinTypes.h"

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
  ControlClass,
  UnknownClass,
};

static std::optional<SchedClassID> getSchedClassForPipe(PIPE pipe) {
  switch (pipe) {
  case PIPE::PIPE_S:
    return ScalarClass;
  case PIPE::PIPE_V:
  case PIPE::PIPE_V2:
    return VectorClass;
  case PIPE::PIPE_M:
    return CubeClass;
  case PIPE::PIPE_MTE1:
  case PIPE::PIPE_MTE2:
  case PIPE::PIPE_MTE3:
  case PIPE::PIPE_MTE4:
  case PIPE::PIPE_MTE5:
  case PIPE::PIPE_FIX:
  case PIPE::VIRTUAL_PIPE_MTE2_L1A:
  case PIPE::VIRTUAL_PIPE_MTE2_L1B:
    return MTEClass;
  case PIPE::PIPE_ALL:
    return ControlClass;
  case PIPE::PIPE_NUM:
  case PIPE::PIPE_UNASSIGNED:
    return std::nullopt;
  }
  return std::nullopt;
}
} // namespace

VPTOGenericA5SchedModel::VPTOGenericA5SchedModel() {
  machine.target = "a5";
  machine.version = "generic-a5-v1";
  machine.issueWidth = 1;
  machine.microOpBufferSize = 0;

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
      {ControlClass, "control", true, 1, 1, {{ControlResource, 0, 1, 1}}, {}},
      {UnknownClass, "unknown", false, 1, 1, {{UnknownResource, 0, 1, 1}}, {}},
  };
}

const VPTOSchedClass &
VPTOGenericA5SchedModel::getSchedClass(Operation *op) const {
  VPTOSchedulingSemantics semantics = getVPTOSchedulingSemantics(op);
  if (!op || semantics.schedulingClass == VPTOSchedulingClass::Structural)
    return schedClasses[StructuralClass];
  if (auto pipeOp = dyn_cast<OpPipeInterface>(op))
    if (std::optional<SchedClassID> schedClass =
            getSchedClassForPipe(pipeOp.getPipe()))
      return schedClasses[*schedClass];
  if (isa<VectorMicroOpInterface>(op))
    return schedClasses[VectorClass];
  if (isa<MteOpInterface>(op))
    return schedClasses[MTEClass];
  if (isa<CubeMicroOpInterface>(op))
    return schedClasses[CubeClass];
  if (isa<SimtOpInterface>(op))
    return schedClasses[ScalarClass];
  if (!semantics.effects.empty())
    return schedClasses[ControlClass];
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

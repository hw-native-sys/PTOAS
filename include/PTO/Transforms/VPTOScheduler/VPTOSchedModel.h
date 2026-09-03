// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedModel.h - VPTO scheduling target model ---------*- C++ -*-===//
//
// The scheduler consumes this read-only contract and does not depend on how a
// target stores its model.  The first implementation is a conservative static
// C++ model; a future TableGen-backed model can implement the same interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDMODEL_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDMODEL_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::pto {

using VPTOSchedResourceID = unsigned;
using VPTOSchedClassID = unsigned;
using VPTOPressureSetID = unsigned;

struct VPTOSchedMachineModel {
  std::string target;
  std::string version;
  unsigned issueWidth = 1;
  unsigned microOpBufferSize = 0;
};

struct VPTOSchedResource {
  VPTOSchedResourceID id = 0;
  std::string name;
  unsigned units = 1;
  unsigned bufferSize = 0;
  SmallVector<VPTOSchedResourceID> groupMembers;
};

struct VPTOSchedResourceUse {
  VPTOSchedResourceID resource = 0;
  unsigned acquireAt = 0;
  unsigned duration = 1;
  unsigned units = 1;
};

struct VPTOSchedClass {
  VPTOSchedClassID id = 0;
  std::string name;
  bool known = false;
  unsigned microOps = 1;
  unsigned writeLatency = 1;
  SmallVector<VPTOSchedResourceUse> resources;
  SmallVector<int> readAdvance;
};

/// Effective scheduling parameters for one operation. Target models may
/// inherit these values from the operation's scheduling class and override
/// individual fields without creating an opcode-specific class.
struct VPTOSchedParameters {
  unsigned microOps = 1;
  unsigned writeLatency = 1;
  ArrayRef<VPTOSchedResourceUse> resources;
  ArrayRef<int> readAdvance;
};

struct VPTORegPressureSet {
  VPTOPressureSetID id = 0;
  std::string name;
  std::optional<unsigned> limit;
  int64_t weight = 1;
  int64_t spillCost = 1;
};

struct VPTORegPressureContribution {
  VPTOPressureSetID pressureSet = 0;
  unsigned units = 0;
};

class VPTOSchedModel {
public:
  virtual ~VPTOSchedModel() = default;

  virtual const VPTOSchedMachineModel &getMachineModel() const = 0;
  virtual ArrayRef<VPTOSchedResource> getResources() const = 0;
  virtual ArrayRef<VPTORegPressureSet> getPressureSets() const = 0;
  virtual const VPTOSchedClass &getSchedClass(Operation *op) const = 0;
  virtual VPTOSchedParameters getSchedParameters(Operation *op) const {
    const VPTOSchedClass &schedClass = getSchedClass(op);
    return {schedClass.microOps, schedClass.writeLatency,
            schedClass.resources, schedClass.readAdvance};
  }
  /// Return the direct SSA source whose physical register pressure is shared
  /// by `value`. Trackers may follow this relation across view-like operations
  /// that belong to their scheduling region.
  virtual Value getPressureRepresentative(Value value) const { return value; }
  virtual SmallVector<VPTORegPressureContribution>
  getPressure(Value value) const = 0;
};

/// Conservative A5 model shared by analyze and on modes. Operations use the
/// generic class for their declared execution pipe or micro-op family, with
/// narrow operation-specific overrides for parameters that differ physically.
class VPTOGenericA5SchedModel final : public VPTOSchedModel {
public:
  VPTOGenericA5SchedModel();

  const VPTOSchedMachineModel &getMachineModel() const override {
    return machine;
  }
  ArrayRef<VPTOSchedResource> getResources() const override {
    return resources;
  }
  ArrayRef<VPTORegPressureSet> getPressureSets() const override {
    return pressureSets;
  }
  const VPTOSchedClass &getSchedClass(Operation *op) const override;
  VPTOSchedParameters getSchedParameters(Operation *op) const override;
  Value getPressureRepresentative(Value value) const override;
  SmallVector<VPTORegPressureContribution>
  getPressure(Value value) const override;

private:
  VPTOSchedMachineModel machine;
  SmallVector<VPTOSchedResource> resources;
  SmallVector<VPTORegPressureSet> pressureSets;
  SmallVector<VPTOSchedClass> schedClasses;
};

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDMODEL_H

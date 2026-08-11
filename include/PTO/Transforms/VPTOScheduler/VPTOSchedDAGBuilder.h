// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOSchedDAGBuilder.h - VPTO scheduling DAG builder -----*- C++ -*-===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDDAGBUILDER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDDAGBUILDER_H

#include "PTO/Transforms/VPTOScheduler/VPTOSchedDAG.h"
#include "PTO/Transforms/VPTOScheduler/VPTOSchedModel.h"

#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir::pto {

class VPTOSchedDAGBuilder {
public:
  explicit VPTOSchedDAGBuilder(const VPTOSchedModel *model = nullptr)
      : model(model) {}

  FailureOr<std::unique_ptr<VPTOSchedDAG>>
  build(const VPTOSchedRegion &region) const;

private:
  void buildSSAEdges(VPTOSchedDAG &dag) const;
  void buildMemoryEdges(VPTOSchedDAG &dag) const;
  void buildImplicitAndSyncEdges(VPTOSchedDAG &dag) const;
  void buildModelFallbackEdges(VPTOSchedDAG &dag) const;

  const VPTOSchedModel *model;
};

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOSCHEDULER_VPTOSCHEDDAGBUILDER_H

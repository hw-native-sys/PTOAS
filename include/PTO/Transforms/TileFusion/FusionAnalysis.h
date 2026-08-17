// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_TILEFUSION_FUSIONANALYSIS_H
#define PTO_TRANSFORMS_TILEFUSION_FUSIONANALYSIS_H

#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/TileFusion/FusionOpSemantics.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

#include <cassert>
#include <optional>

namespace mlir {
namespace pto {

enum class IterationDomainProof {
  Proven,
  Unproven,
};

enum class IterationDomainUnprovenReason {
  None,
  MissingTileDomain,
  DynamicShape,
  InconsistentShape,
};

struct IterationDomainInfo {
  int64_t vRow = ShapedType::kDynamic;
  int64_t vCol = ShapedType::kDynamic;
  IterationDomainProof proof = IterationDomainProof::Unproven;
  IterationDomainUnprovenReason unprovenReason =
      IterationDomainUnprovenReason::MissingTileDomain;
};

struct IterationDomainClass {
  unsigned id = 0;
  IterationDomainInfo info;
  SmallVector<unsigned, mlir::pto::kValue4> members;
};

struct FusionDFGEdge {
  unsigned producerNode = 0;
  unsigned consumerNode = 0;
  Value value;
};

struct FusionValueLiveness {
  Value value;
  std::optional<unsigned> producerNode;
  SmallVector<unsigned, mlir::pto::kValue4> consumerNodes;
  SmallVector<unsigned, mlir::pto::kValue2> writeInstances;
  std::optional<unsigned> lastLocalConsumer;
  bool hasExternalUsers = false;
  bool escapesBlock = false;
  bool hasLocalBoundaryUsers = false;
  bool hasLocalHardBoundaryUsers = false;
};

enum class FusionWriteInstanceEscapeClass {
  Internal,
  LocalBoundaryExternal,
  HardExternal,
};

struct FusionWriteInstanceLiveness {
  unsigned id = 0;
  Value value;
  Value storageValue;
  std::optional<unsigned> producerNode;
  SmallVector<unsigned, mlir::pto::kValue4> consumerNodes;
  std::optional<unsigned> lastLocalConsumer;
  FusionWriteInstanceEscapeClass escapeClass =
      FusionWriteInstanceEscapeClass::Internal;
  bool hasExternalUsers = false;
  bool escapesBlock = false;
  bool hasLocalBoundaryUsers = false;
  bool hasLocalHardBoundaryUsers = false;
};

struct FusionComputeNode {
  unsigned id = 0;
  unsigned blockOrder = 0;
  Operation *op = nullptr;
  FusionOpSemantics semantics;
  unsigned iterationDomainClass = 0;
  SmallVector<unsigned, mlir::pto::kValue4> incomingEdges;
  SmallVector<unsigned, mlir::pto::kValue4> outgoingEdges;
};

struct FusionBlockAnalysis {
  Block *block = nullptr;
  SmallVector<FusionComputeNode, mlir::pto::kValue8> computeNodes;
  SmallVector<IterationDomainClass, mlir::pto::kValue4> iterationDomainClasses;
  SmallVector<FusionDFGEdge, mlir::pto::kValue8> edges;
  SmallVector<FusionValueLiveness, mlir::pto::kValue8> liveness;
  SmallVector<FusionWriteInstanceLiveness, mlir::pto::kValue8> writeInstances;
};

struct PreFusionAnalysisResult {
  SmallVector<FusionBlockAnalysis, mlir::pto::kValue8> blocks;
};

/// Build the *shared* pre-fusion analysis: the tile dataflow graph
/// (compute nodes, DFG edges, value liveness, write instances) for every block
/// of \p func, WITHOUT inferring iteration-domain classes.  This is the
/// expensive part that walks the whole IR and builds the alias/liveness
/// structures, and it is identical regardless of whether shape inference is
/// enabled.  Iteration-domain inference (see inferIterationDomainClasses) is a
/// cheap, separable follow-up step that only the FusionPlan pass needs.
///
/// This function is intentionally option-free so it can be cached by the MLIR
/// analysis manager (PreFusionAnalysis) and shared by every pass that only
/// consumes the dataflow structures (e.g. FusionRegionGen).
FailureOr<PreFusionAnalysisResult>
buildPreFusionAnalysisDFG(func::FuncOp func);

/// Infer iteration-domain classes for every block of \p result.  When
/// \p enableShapeInference is false, uses the conservative static/direct-bound
/// inference; when true, uses the ShapeConstraintSolver.  This is the separable
/// domain step that runs on top of the shared dataflow graph built by
/// buildPreFusionAnalysisDFG.
LogicalResult inferIterationDomainClasses(PreFusionAnalysisResult &result,
                                         bool enableShapeInference);

/// Convenience: build the shared DFG and infer iteration-domain classes in one
/// step.  Prefer calling buildPreFusionAnalysisDFG + inferIterationDomainClasses
/// separately when the DFG can be shared across passes via the analysis
/// manager.
FailureOr<PreFusionAnalysisResult>
buildPreFusionAnalysis(func::FuncOp func, bool enableShapeInference = false);

/// Analysis-manager-cached pre-fusion analysis.  Holds only the shared
/// dataflow graph (no iteration-domain classes): iteration-domain inference
/// depends on the per-pass --enable-shape-inference option and cannot be cached
/// here, so it is run separately by FusionPlan via inferIterationDomainClasses.
/// FusionRegionGen consumes only the dataflow structures and never consults the
/// domain classes, so it can use the cached graph directly.
class PreFusionAnalysis {
public:
  explicit PreFusionAnalysis(func::FuncOp func) {
    FailureOr<PreFusionAnalysisResult> resultOr =
        buildPreFusionAnalysisDFG(func);
    if (succeeded(resultOr)) {
      result = std::move(*resultOr);
    }
  }

  bool isValid() const { return result.has_value(); }

  const PreFusionAnalysisResult &getResult() const {
    assert(result && "expected valid pre-fusion analysis result");
    return *result;
  }

private:
  std::optional<PreFusionAnalysisResult> result;
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_TILEFUSION_FUSIONANALYSIS_H

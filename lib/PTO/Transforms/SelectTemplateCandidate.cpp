// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/TileShapeStateAnalysis.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringSwitch.h"

#include <optional>

using namespace mlir;

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_SELECTTEMPLATECANDIDATE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

constexpr llvm::StringLiteral kCandidatesAttr = "candidates";
constexpr llvm::StringLiteral kSelectedCandidateAttr =
    "pto.tilelib.selected_candidate";
constexpr llvm::StringLiteral kTileLibImplAttr = "pto.tilelib.impl";
constexpr llvm::StringLiteral kVmiFusionBoundaryAttr =
    "pto.vmi.fusion.boundary";
constexpr llvm::StringLiteral kVmiFusionBoundaryReasonAttr =
    "pto.vmi.fusion.boundary_reason";
constexpr llvm::StringLiteral kVmiEstimatedPeakVectorBytesAttr =
    "pto.vmi.resource.estimated_peak_vector_bytes";
constexpr llvm::StringLiteral kVmiEstimatedPeakVectorChunksAttr =
    "pto.vmi.resource.estimated_peak_vector_chunks";
constexpr llvm::StringLiteral kVmiResourceEstimateExactAttr =
    "pto.vmi.resource.estimate_exact";

constexpr int64_t kA5PhysicalVectorBytes = 256;

struct VMIResourceEstimate {
  int64_t peakVectorBytes = 0;
  int64_t peakVectorChunks = 0;
  bool isExact = false;
  StringRef rejectionReason = "resource_estimate_unknown";
};

static bool candidateHasTag(DictionaryAttr candidate, StringRef tag) {
  auto tags = candidate.getAs<ArrayAttr>("tags");
  if (!tags)
    return false;
  return llvm::any_of(tags, [tag](Attribute attr) {
    auto value = dyn_cast<StringAttr>(attr);
    return value && value.getValue() == tag;
  });
}

static StringRef getTileOpName(Operation *op) {
  return op->getName().getStringRef().split('.').second;
}

static bool isHardBoundaryFallbackOp(Operation *op) {
  auto pipeOp = dyn_cast<pto::OpPipeInterface>(op);
  if (!pipeOp || pipeOp.getPipe() != pto::PIPE::PIPE_V)
    return true;
  return llvm::StringSwitch<bool>(getTileOpName(op))
      .Cases("tload", "tstore", "tmatmul", "tmatmul_acc", "tmatmul_bias",
             "tmatmul_mx", true)
      .Cases("tmrgsort", "tsort32", "tpush", "tpop", "tfree", true)
      .Cases("tgather", "tgatherb", "tscatter", "tscatterb", true)
      .Cases("textract", "textract_fp", "tfillpad", "tfillpad_expand",
             "tfillpad_inplace", true)
      .Cases("tconcat", "tinsert", "tci", true)
      .Default(false);
}

static bool isHardBoundaryFallback(Operation *op, DictionaryAttr candidate) {
  return isHardBoundaryFallbackOp(op) ||
         candidateHasTag(candidate, "hard_boundary");
}

static bool hasMaterializationSensitiveSubview(Operation *op) {
  return llvm::any_of(op->getOperands(), [](Value operand) {
    auto subview = operand.getDefiningOp<pto::SubViewOp>();
    if (!subview)
      return false;
    auto sourceType = dyn_cast<pto::TileBufType>(subview.getSource().getType());
    auto resultType = dyn_cast<pto::TileBufType>(operand.getType());
    if (!sourceType || !resultType)
      return false;

    // Materialization preserves the parent's physical extent for a narrowed
    // subview. Pre-materialization shape metadata cannot prove such a VMI
    // candidate legal unless the candidate opts into this form.
    return sourceType.getShape() != resultType.getShape();
  });
}

static std::optional<int64_t> getElementBytes(Type type) {
  if (type.isF32() || type.isInteger(32))
    return 4;
  if (type.isF16() || type.isBF16() || type.isInteger(16))
    return 2;
  if (type.isInteger(8) || type.isInteger(1))
    return 1;
  if (type.isF64() || type.isInteger(64))
    return 8;
  return std::nullopt;
}

static std::optional<int64_t> roundToPhysicalVectorBytes(int64_t bytes) {
  int64_t rounded = 0;
  if (bytes <= 0 ||
      __builtin_add_overflow(bytes, kA5PhysicalVectorBytes - 1, &rounded))
    return std::nullopt;
  return rounded / kA5PhysicalVectorBytes * kA5PhysicalVectorBytes;
}

static std::optional<int64_t> getMaterializedVectorBytes(pto::TileBufType type,
                                                         StringRef scope) {
  ArrayRef<int64_t> shape = type.getShape();
  if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0)
    return std::nullopt;
  std::optional<int64_t> elementBytes = getElementBytes(type.getElementType());
  if (!elementBytes)
    return std::nullopt;

  int64_t elements = shape[1];
  if (scope == "tile" && __builtin_mul_overflow(elements, shape[0], &elements))
    return std::nullopt;
  int64_t bytes = 0;
  if (__builtin_mul_overflow(elements, *elementBytes, &bytes))
    return std::nullopt;
  return roundToPhysicalVectorBytes(bytes);
}

/// Estimate the peak bytes represented by simultaneously materialized logical
/// VMI vectors. This is a conservative candidate-selection guard, not a claim
/// about the number of A5 physical registers. Contracts are emitted by the
/// TileLib provider and remain explicit in the selected-candidate metadata.
static VMIResourceEstimate estimateCandidateResource(Operation *op,
                                                     DictionaryAttr candidate) {
  VMIResourceEstimate estimate;
  auto scopeAttr = candidate.getAs<StringAttr>("resource_scope");
  auto valueCountAttr = candidate.getAs<IntegerAttr>("resource_vector_values");
  auto chunkStreamingAttr =
      candidate.getAs<BoolAttr>("resource_chunk_streaming");
  if (!scopeAttr || !valueCountAttr || !chunkStreamingAttr ||
      (scopeAttr.getValue() != "row" && scopeAttr.getValue() != "tile") ||
      valueCountAttr.getInt() <= 0)
    return estimate;

  int64_t materializedBytes = 0;
  bool sawTile = false;
  for (Value operand : op->getOperands()) {
    auto tile = dyn_cast<pto::TileBufType>(operand.getType());
    if (!tile)
      continue;
    sawTile = true;
    std::optional<int64_t> bytes =
        getMaterializedVectorBytes(tile, scopeAttr.getValue());
    if (!bytes)
      return estimate;
    materializedBytes = std::max(materializedBytes, *bytes);
  }
  if (!sawTile)
    return estimate;

  if (chunkStreamingAttr.getValue())
    materializedBytes = kA5PhysicalVectorBytes;
  if (__builtin_mul_overflow(materializedBytes, valueCountAttr.getInt(),
                             &estimate.peakVectorBytes))
    return estimate;
  estimate.peakVectorChunks = estimate.peakVectorBytes / kA5PhysicalVectorBytes;
  estimate.isExact = true;
  estimate.rejectionReason = "resource_pressure_fallback";
  return estimate;
}

static void clearResourceAttrs(Operation *op) {
  op->removeAttr(kVmiEstimatedPeakVectorBytesAttr);
  op->removeAttr(kVmiEstimatedPeakVectorChunksAttr);
  op->removeAttr(kVmiResourceEstimateExactAttr);
}

static void recordSelection(Operation *op, DictionaryAttr candidate, bool isVMI,
                            const VMIResourceEstimate *estimate = nullptr,
                            StringRef fallbackReason = "") {
  op->setAttr(kSelectedCandidateAttr, candidate);
  op->setAttr(kTileLibImplAttr,
              StringAttr::get(op->getContext(), isVMI ? "vmi" : "ptodsl"));
  op->removeAttr(kVmiFusionBoundaryAttr);
  op->removeAttr(kVmiFusionBoundaryReasonAttr);
  clearResourceAttrs(op);

  Builder builder(op->getContext());
  if (estimate) {
    op->setAttr(kVmiResourceEstimateExactAttr,
                builder.getBoolAttr(estimate->isExact));
    if (estimate->isExact) {
      op->setAttr(kVmiEstimatedPeakVectorBytesAttr,
                  builder.getI64IntegerAttr(estimate->peakVectorBytes));
      op->setAttr(kVmiEstimatedPeakVectorChunksAttr,
                  builder.getI64IntegerAttr(estimate->peakVectorChunks));
    }
  }
  if (isVMI && candidateHasTag(candidate, "fusion_eligible"))
    return;

  bool hard = !isVMI && isHardBoundaryFallback(op, candidate);
  op->setAttr(kVmiFusionBoundaryAttr,
              StringAttr::get(op->getContext(), hard ? "hard" : "local"));
  StringRef reason = fallbackReason;
  if (isVMI)
    reason = "vmi_non_fusion_eligible_candidate";
  else if (reason.empty())
    reason = hard ? "non_vmi_hard_boundary_fallback"
                  : "non_vmi_local_boundary_fallback";
  op->setAttr(kVmiFusionBoundaryReasonAttr,
              StringAttr::get(op->getContext(), reason));
}

static bool isLegalVMIChoice(Operation *op, DictionaryAttr candidate,
                             bool hardBoundary) {
  return !hardBoundary && candidateHasTag(candidate, "vmi") &&
         (pto::hasStaticFullTileValidShape(op) ||
          candidateHasTag(candidate, "supports_partial_valid_shape")) &&
         (!hasMaterializationSensitiveSubview(op) ||
          candidateHasTag(candidate,
                          "supports_materialization_sensitive_subview"));
}

struct SelectTemplateCandidatePass
    : pto::impl::SelectTemplateCandidateBase<SelectTemplateCandidatePass> {
  using SelectTemplateCandidateBase::SelectTemplateCandidateBase;

  void runOnOperation() override {
    if (selectionPolicy != "prefer-vmi" && selectionPolicy != "ordinary-only") {
      getOperation().emitError("unsupported template selection policy '")
          << selectionPolicy << "'";
      return signalPassFailure();
    }
    if (maxCandidateVectorBytes < 0) {
      getOperation().emitError(
          "max-candidate-vector-bytes must be non-negative");
      return signalPassFailure();
    }

    WalkResult result = getOperation().walk([&](Operation *op) {
      auto candidates = op->getAttrOfType<ArrayAttr>(kCandidatesAttr);
      if (!candidates)
        return WalkResult::advance();

      SmallVector<DictionaryAttr> parsed;
      DictionaryAttr ordinary;
      for (Attribute attr : candidates) {
        auto candidate = dyn_cast<DictionaryAttr>(attr);
        if (!candidate || !candidate.getAs<StringAttr>("name")) {
          op->emitError(
              "template candidate must be a dictionary with a string name");
          return WalkResult::interrupt();
        }
        parsed.push_back(candidate);
        if (!candidateHasTag(candidate, "vmi") && !ordinary)
          ordinary = candidate;
      }
      if (!ordinary) {
        // When no ordinary (non-vmi) fallback candidate exists, the VMI
        // selection loop below may still select a legal vmi candidate. Only
        // fail hard when there are no candidates at all.
        if (parsed.empty()) {
          op->emitError(
              "no PTODSL TileLib candidate available");
          return WalkResult::interrupt();
        }
      }

      const bool hardBoundary =
          isHardBoundaryFallbackOp(op) ||
          llvm::any_of(parsed, [](DictionaryAttr candidate) {
            return candidateHasTag(candidate, "hard_boundary");
          });
      std::optional<VMIResourceEstimate> preferredRejectedEstimate;
      if (selectionPolicy == "prefer-vmi") {
        auto trySelect = [&](DictionaryAttr candidate) {
          if (!isLegalVMIChoice(op, candidate, hardBoundary))
            return false;
          VMIResourceEstimate estimate =
              estimateCandidateResource(op, candidate);
          const bool guardDisabled = maxCandidateVectorBytes == 0;
          const bool withinBudget =
              estimate.isExact &&
              estimate.peakVectorBytes <= maxCandidateVectorBytes;
          if (guardDisabled || withinBudget) {
            recordSelection(op, candidate, true, &estimate);
            if (emitResourceRemarks) {
              auto name = candidate.getAs<StringAttr>("name").getValue();
              if (estimate.isExact)
                op->emitRemark()
                    << "VMI candidate '" << name
                    << "' accepted with estimated peak "
                    << estimate.peakVectorBytes << " vector bytes ("
                    << estimate.peakVectorChunks << " chunks)";
              else
                op->emitRemark()
                    << "VMI candidate '" << name
                    << "' accepted because the resource guard is disabled";
            }
            return true;
          }

          if (!preferredRejectedEstimate)
            preferredRejectedEstimate = estimate;
          if (emitResourceRemarks) {
            auto name = candidate.getAs<StringAttr>("name").getValue();
            if (estimate.isExact)
              op->emitRemark()
                  << "VMI candidate '" << name << "' rejected: estimated peak "
                  << estimate.peakVectorBytes << " vector bytes exceeds "
                  << static_cast<int64_t>(maxCandidateVectorBytes);
            else
              op->emitRemark() << "VMI candidate '" << name
                               << "' rejected: resource contract is missing "
                                  "or cannot be evaluated";
          }
          return false;
        };

        for (DictionaryAttr candidate : parsed) {
          if (candidateHasTag(candidate, "row_streaming") &&
              candidateHasTag(candidate, "single_logical_row_loop") &&
              trySelect(candidate))
            return WalkResult::advance();
        }
        for (DictionaryAttr candidate : parsed) {
          if (!candidateHasTag(candidate, "row_streaming") &&
              candidateHasTag(candidate, "fusion_eligible") &&
              trySelect(candidate))
            return WalkResult::advance();
        }
        for (DictionaryAttr candidate : parsed) {
          if (!candidateHasTag(candidate, "fusion_eligible") &&
              trySelect(candidate))
            return WalkResult::advance();
        }
      }

      if (preferredRejectedEstimate) {
        if (ordinary) {
          recordSelection(op, ordinary, false, &*preferredRejectedEstimate,
                          preferredRejectedEstimate->rejectionReason);
          return WalkResult::advance();
        }
        op->emitError("VMI candidate rejected and no ordinary fallback "
                      "available");
        return WalkResult::interrupt();
      }
      if (ordinary) {
        recordSelection(op, ordinary, false);
        return WalkResult::advance();
      }
      op->emitError("no legal VMI candidate and no ordinary fallback "
                    "available");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace pto {
std::unique_ptr<Pass> createSelectTemplateCandidatePass() {
  return std::make_unique<SelectTemplateCandidatePass>();
}
std::unique_ptr<Pass> createSelectTemplateCandidatePass(
    const SelectTemplateCandidateOptions &options) {
  return std::make_unique<SelectTemplateCandidatePass>(options);
}
} // namespace pto
} // namespace mlir

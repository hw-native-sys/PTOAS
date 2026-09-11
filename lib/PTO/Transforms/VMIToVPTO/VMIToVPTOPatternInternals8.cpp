// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals8.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

static LogicalResult checkVmullPhysicalShape(VMIVRegType dataType,
                                             VMIMaskType maskType,
                                             std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(dataType.getElementType());
  Type physicalElementType = getVMIPhysicalDataElementType(dataType);
  FailureOr<StringRef> physicalMaskGranularity =
      getVMIMaskPhysicalGranularity(maskType);
  bool invalidPhysicalShape =
      failed(lanesPerPart) || *lanesPerPart != 64 ||
      physicalElementType != dataType.getElementType() ||
      failed(physicalMaskGranularity) || *physicalMaskGranularity != "b32";
  if (invalidPhysicalShape) {
    return fail("requires 64xi32/ui32 data parts with corresponding b32 mask "
                "parts");
  }
  return success();
}

LogicalResult checkSupportedVmullShape(VMIVmullOp op,
                                       std::string *reason = nullptr) {
  FailureOr<VmullShapePlan> plan = buildVmullShapePlan(op, reason);
  if (failed(plan)) {
    return failure();
  }
  VMIVRegType aType = plan->dataType;
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  return checkVmullPhysicalShape(aType, maskType, reason);
}

static LogicalResult checkAddCarryMaskPort(VMIMaskType maskType,
                                           VMILayoutAttr dataLayout,
                                           int64_t dataArity,
                                           std::string *reason) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };
  bool layoutMismatch = maskType.getLayoutAttr() != dataLayout;
  if (layoutMismatch) {
    return fail("requires all data and mask ports to share one layout");
  }
  bool unsupportedGranularity = maskType.getGranularity() != "b32";
  if (unsupportedGranularity) {
    return fail("requires b32 mask granularity");
  }
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  bool hasMatchingArity = succeeded(maskArity) && *maskArity == dataArity;
  if (!hasMatchingArity) {
    return fail("requires matching physical arity on data and mask ports");
  }
  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(maskType);
  bool unsupportedPhysicalGranularity = failed(physicalGranularity) ||
                                        *physicalGranularity != "b32";
  if (unsupportedPhysicalGranularity) {
    return fail("requires physical b32 mask parts");
  }
  return success();
}

static LogicalResult
checkSupportedVMIAddCarryPorts(VMIVRegType lhsType, VMIVRegType rhsType,
                               VMIVRegType resultType,
                               ArrayRef<VMIMaskType> maskTypes,
                               std::string *reason = nullptr) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto integerType = dyn_cast<IntegerType>(lhsType.getElementType());
  bool unsupportedIntegerType = !integerType || integerType.getWidth() != 32;
  if (unsupportedIntegerType) {
    return fail("requires 32-bit integer data elements");
  }
  bool mismatchedTypes = lhsType != rhsType || lhsType != resultType;
  if (mismatchedTypes) {
    return fail("requires matching lhs, rhs, and result VMI types");
  }
  if (!lhsType.getLayoutAttr()) {
    return fail("requires assigned data layout");
  }
  if (failed(checkSupportedMaskableVReg(lhsType))) {
    return fail("requires computable physical data parts");
  }

  FailureOr<int64_t> dataArity = getVMIPhysicalArity(lhsType);
  bool invalidDataArity = failed(dataArity) || *dataArity < 1;
  if (invalidDataArity) {
    return fail("requires non-empty physical data parts");
  }
  for (VMIMaskType maskType : maskTypes) {
    if (failed(checkAddCarryMaskPort(maskType, lhsType.getLayoutAttr(),
                                     *dataArity, reason))) {
      return failure();
    }
  }
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(lhsType.getElementType());
  bool hasExpectedLanes = succeeded(lanesPerPart) && *lanesPerPart == 64;
  if (!hasExpectedLanes) {
    return fail("requires 64-lane 32-bit data parts");
  }
  return success();
}

LogicalResult checkSupportedVMIAddcShape(VMIVaddcOp op,
                                         std::string *reason = nullptr) {
  return checkSupportedVMIAddCarryPorts(
      cast<VMIVRegType>(op.getLhs().getType()),
      cast<VMIVRegType>(op.getRhs().getType()),
      cast<VMIVRegType>(op.getResult().getType()),
      {cast<VMIMaskType>(op.getMask().getType()),
       cast<VMIMaskType>(op.getCarry().getType())},
      reason);
}

LogicalResult checkSupportedVMIAddcsShape(VMIVaddcsOp op,
                                          std::string *reason = nullptr) {
  return checkSupportedVMIAddCarryPorts(
      cast<VMIVRegType>(op.getLhs().getType()),
      cast<VMIVRegType>(op.getRhs().getType()),
      cast<VMIVRegType>(op.getResult().getType()),
      {cast<VMIMaskType>(op.getCarryIn().getType()),
       cast<VMIMaskType>(op.getMask().getType()),
       cast<VMIMaskType>(op.getCarry().getType())},
      reason);
}

LogicalResult
checkSupportedFmaShape(VMIFmaOp op, std::string *reason = nullptr) {
  auto fail = [&reason](const Twine &message) -> LogicalResult {
    if (reason) {
      *reason = message.str();
    }
    return failure();
  };

  auto lhsType = cast<VMIVRegType>(op.getLhs().getType());
  FailureOr<int64_t> arity = getVMIPhysicalArity(lhsType);
  bool hasNonEmptyArity = succeeded(arity) && *arity >= 1;
  if (!hasNonEmptyArity) {
    return fail("requires computable non-empty physical arity");
  }

  return success();
}

LogicalResult
checkSupportedReluShape(VMIReluOp op, std::string *reason = nullptr) {
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (failed(checkSupportedMaskableVReg(resultType, reason))) {
    return failure();
  }

  return success();
}

LogicalResult
checkSupportedVselrShape(VMIVselrOp op, std::string *reason = nullptr) {
  VMILayoutSupport supports;
  return supports.getVselrSupport(op, reason);
}

void emitEnsureLayoutMaterializationError(VMIEnsureLayoutOp ensure,
                                          VMIVRegType sourceType,
                                          VMIVRegType resultType,
                                          StringRef reason) {
  if (ensure.getResult().hasOneUse()) {
    OpOperand &use = *ensure.getResult().use_begin();
    Operation *requester = use.getOwner();
    InFlightDiagnostic diag =
        requester->emitError()
        << kVMIDiagUnsupportedPrefix << requester->getName() << " operand #"
        << use.getOperandNumber() << " has type " << sourceType
        << " but requires " << resultType
        << "; pto.vmi.ensure_layout cannot materialize this conversion";
    diag.attachNote(ensure.getLoc())
        << "failed helper conversion " << sourceType << " -> " << resultType
        << " (" << reason
        << "); partial/tail layout materialization requires an explicit "
           "packing plan";
    return;
  }

  ensure.emitError()
      << kVMIDiagUnsupportedPrefix
      << "pto.vmi.ensure_layout cannot materialize the requested data "
         "layout conversion ("
      << reason
      << "); partial/tail layout materialization requires an explicit "
         "packing plan";
}

WalkResult emitMemoryUnsupported(Operation *memoryOp, StringRef opName,
                                 VMIVRegType type, Value source,
                                 std::optional<int64_t> constantOffset) {
    std::string reason;
    if (succeeded(checkSupportedLoadShape(type, source, source.getType(),
                                          constantOffset, &reason))) {
      return WalkResult::advance();
    }

    memoryOp->emitError()
        << kVMIDiagUnsupportedPrefix << opName
        << " direct lowering requires a supported memory source (" << reason
        << ")";
    return WalkResult::interrupt();
}

static std::optional<WalkResult> verifySupportedVMIMaskedLoadOp(
    Operation *op, bool enableStableGatherMaskedLoad) {
  if (auto load = dyn_cast<VMIMaskedLoadOp>(op)) {
    if (enableStableGatherMaskedLoad) {
      load.emitError() << kVMIDiagUnsupportedPrefix
                       << "pto.vmi.masked_load stable VGATHER-based lowering "
                          "is reserved for strict masked/tail loads but is "
                          "not implemented yet";
      return WalkResult::interrupt();
    }
    return verifySupportedShapeOp(
        load, checkSupportedMaskedLoadShape,
        "pto.vmi.masked_load direct lowering requires a supported memory source, "
        "contiguous result/passthru/mask layouts, and either full physical "
        "chunks or a statically safe full-read footprint (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIGatherOp(Operation *op) {
  if (auto gather = dyn_cast<VMIGatherOp>(op)) {
    return verifySupportedShapeOp(
        gather, checkSupportedGatherShape,
        "pto.vmi.gather lowers through pto.vgather2/pto.vgather2_bc + pto.vsel "
        "only for UB pointer sources, contiguous full physical chunks, "
        "ui16/i16/f16/bf16 results with ui16 indices and b16 masks, or "
        "32-bit results with i32 indices and b32 masks (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIExpandLoadOp(Operation *op) {
  if (auto load = dyn_cast<VMIExpandLoadOp>(op)) {
    return verifySupportedShapeOp(
        load, checkSupportedExpandLoadShape,
        "pto.vmi.expand_load direct lowering is currently supported for either "
        "a static all-active mask lowered as pto.vlds, or a one-full-chunk "
        "32-bit UB runtime mask lowered through pto.vusqz + pto.vgather2_bc + "
        "pto.vsel (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIMemoryAdvancedLoadOp(
    Operation *op, bool enableStableGatherMaskedLoad) {
  if (auto result = verifySupportedVMIMaskedLoadOp(
          op, enableStableGatherMaskedLoad);
      result.has_value()) {
    return result;
  }
  if (auto result = verifySupportedVMIGatherOp(op); result.has_value()) {
    return result;
  }
  return verifySupportedVMIExpandLoadOp(op);
}

static std::optional<WalkResult> verifySupportedVMIStructuredMaskedStoreOp(
    Operation *op) {
  if (auto store = dyn_cast<VMIMaskedStoreOp>(op)) {
    std::string reason;
    if (succeeded(checkSupportedMaskedStoreShape(
            cast<VMIVRegType>(store.getValue().getType()),
            cast<VMIMaskType>(store.getMask().getType()),
            store.getDestination(), store.getDestination().getType(),
            &reason))) {
      return WalkResult::advance();
    }
    store.emitError()
        << kVMIDiagUnsupportedPrefix
        << "pto.vmi.masked_store requires either full physical chunks or "
           "contiguous tail-store value/mask layout, with UB-backed "
           "destination ("
        << reason << ")";
    return WalkResult::interrupt();
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIInterleaveStoreOp(
    Operation *op) {
  if (auto store = dyn_cast<VMIInterleaveStoreOp>(op)) {
    return verifySupportedShapeOp(
        store, checkSupportedInterleaveStoreShape,
        "pto.vmi.interleave_store lowers through pto.vstsx2 only for matching "
        "contiguous full low/high input chunks with a supported UB destination "
        "and 8/16/32-bit element type (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIGroupStoreOp(Operation *op) {
  if (auto store = dyn_cast<VMIGroupStoreOp>(op)) {
    return verifySupportedShapeOp(
        store, checkSupportedGroupStoreShape,
        "pto.vmi.group_store requires a supported UB destination and a table-"
        "supported value layout lowering through one-block vsstb, full-chunk "
        "vsts, or deinterleaved vstsx2 (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIStrideStoreOp(Operation *op) {
  if (auto store = dyn_cast<VMIStrideStoreOp>(op)) {
    return verifySupportedShapeOp(
        store, checkSupportedStrideStoreShape,
        "pto.vmi.stride_store lowers through pto.vsstb only for one contiguous "
        "physical value/mask chunk and a supported UB destination (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIScatterOp(Operation *op) {
  if (auto scatter = dyn_cast<VMIScatterOp>(op)) {
    return verifySupportedShapeOp(
        scatter, checkSupportedScatterShape,
        "pto.vmi.scatter lowers through pto.vscatter only with a UB pointer "
        "destination, contiguous full physical chunks, 32-bit value elements, "
        "i32 indices, and b32 masks (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIStructuredStoreOp(Operation *op) {
  if (auto maskedStore = verifySupportedVMIStructuredMaskedStoreOp(op);
      maskedStore.has_value()) {
    return maskedStore;
  }
  if (auto result = verifySupportedVMIInterleaveStoreOp(op);
      result.has_value()) {
    return result;
  }
  if (auto result = verifySupportedVMIGroupStoreOp(op); result.has_value()) {
    return result;
  }
  if (auto result = verifySupportedVMIStrideStoreOp(op); result.has_value()) {
    return result;
  }
  return verifySupportedVMIScatterOp(op);
}

static LogicalResult checkSupportedVMIStoreShape(VMIStoreOp op,
                                                 std::string *reason) {
  return checkSupportedStoreShape(
      cast<VMIVRegType>(op.getValue().getType()), op.getDestination(),
      op.getDestination().getType(), reason);
}

std::optional<WalkResult> verifySupportedVMIMemoryStoreOp(Operation *op) {
  if (auto store = dyn_cast<VMIStoreOp>(op)) {
    return verifySupportedShapeOp(
        store, checkSupportedVMIStoreShape,
        "pto.vmi.store requires an 8/16/32-bit predicate-maskable element "
        "type and either full physical chunks or contiguous tail-store "
        "layout, with UB-backed destination (");
  }
  if (auto structuredResult = verifySupportedVMIStructuredStoreOp(op);
      structuredResult.has_value()) {
    return *structuredResult;
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIDeinterleaveLoadOp(
    Operation *op) {
  if (auto load = dyn_cast<VMIDeinterleaveLoadOp>(op)) {
    return verifySupportedShapeOp(
        load, checkSupportedDeinterleaveLoadShape,
        "pto.vmi.deinterleave_load lowers through pto.vldsx2 only for "
        "matching contiguous full low/high result chunks with a supported "
        "UB source and 8/16/32-bit element type (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIStrideLoadOp(Operation *op) {
  if (auto load = dyn_cast<VMIStrideLoadOp>(op)) {
    return verifySupportedShapeOp(
        load, checkSupportedStrideLoadShape,
        "pto.vmi.stride_load lowers through pto.vsldb only for one "
        "contiguous physical result/mask chunk and a supported UB source (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIGroupLoadOp(Operation *op) {
  if (auto load = dyn_cast<VMIGroupLoadOp>(op)) {
    return verifySupportedShapeOp(
        load, checkSupportedGroupLoadShape,
        "pto.vmi.group_load requires contiguous full result chunks, a "
        "supported UB source, and num_groups deriving a group size aligned "
        "to physical chunks (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIGroupSlotLoadOp(
    Operation *op) {
  if (auto load = dyn_cast<VMIGroupSlotLoadOp>(op)) {
    return verifySupportedShapeOp(
        load, checkSupportedGroupSlotLoadShape,
        "pto.vmi.group_slot_load requires explicit group_slots result layout "
        "matching num_groups, a supported UB pointer source, and either "
        "slots=8 with constant unit source_group_stride or slots=1 row-local "
        "lowering (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIGroupBroadcastLoadOp(
    Operation *op) {
  if (auto load = dyn_cast<VMIGroupBroadcastLoadOp>(op)) {
    return verifySupportedShapeOp(
        load, checkSupportedGroupBroadcastLoadShape,
        "pto.vmi.group_broadcast_load requires either the BRC full-group "
        "chunk form, the E2B packet form for b16/b32 direct or split group "
        "size, or the generic group-slot-load then group-broadcast fallback "
        "with supported UB pointer source and source_group_stride (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIStructuredLoadOp(Operation *op) {
  if (auto result = verifySupportedVMIDeinterleaveLoadOp(op);
      result.has_value()) {
    return result;
  }
  if (auto result = verifySupportedVMIStrideLoadOp(op); result.has_value()) {
    return result;
  }
  if (auto result = verifySupportedVMIGroupLoadOp(op); result.has_value()) {
    return result;
  }
  if (auto result = verifySupportedVMIGroupSlotLoadOp(op);
      result.has_value()) {
    return result;
  }
  return verifySupportedVMIGroupBroadcastLoadOp(op);
}

std::optional<WalkResult> verifySupportedVMIMemoryLoadOp(
    Operation *op, bool enableStableGatherMaskedLoad) {
  if (auto load = dyn_cast<VMILoadOp>(op)) {
    return emitMemoryUnsupported(
        op, "pto.vmi.load", cast<VMIVRegType>(load.getResult().getType()),
        load.getSource(), getConstantIndexValue(load.getOffset()));
  }
  if (auto structuredResult = verifySupportedVMIStructuredLoadOp(op);
      structuredResult.has_value()) {
    return *structuredResult;
  }
  if (auto advancedResult = verifySupportedVMIMemoryAdvancedLoadOp(
          op, enableStableGatherMaskedLoad);
      advancedResult.has_value()) {
    return *advancedResult;
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIMemoryOp(
    Operation *op, bool enableStableGatherMaskedLoad) {
  if (auto loadResult = verifySupportedVMIMemoryLoadOp(
          op, enableStableGatherMaskedLoad);
      loadResult.has_value()) {
    return *loadResult;
  }
  return verifySupportedVMIMemoryStoreOp(op);
}

std::optional<WalkResult> verifySupportedVMIEnsureLayoutOp(Operation *op) {
  if (auto ensure = dyn_cast<VMIEnsureLayoutOp>(op)) {
    auto sourceType = cast<VMIVRegType>(ensure.getSource().getType());
    auto resultType = cast<VMIVRegType>(ensure.getResult().getType());
    std::string reason;
    VMILayoutSupport supports;
    if (succeeded(supports.getEnsureLayoutFact(sourceType, resultType,
                                               &reason))) {
      return WalkResult::advance();
    }

    emitEnsureLayoutMaterializationError(ensure, sourceType, resultType,
                                         reason);
    return WalkResult::interrupt();
  }

  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIMaskLayoutOp(Operation *op) {
  if (auto ensure = dyn_cast<VMIEnsureMaskLayoutOp>(op)) {
    auto sourceType = cast<VMIMaskType>(ensure.getSource().getType());
    auto resultType = cast<VMIMaskType>(ensure.getResult().getType());
    std::string reason;
    VMILayoutSupport supports;
    if (succeeded(supports.getEnsureMaskLayoutFact(sourceType, resultType,
                                                   &reason))) {
      return WalkResult::advance();
    }

    ensure.emitError()
        << kVMIDiagUnsupportedPrefix
        << "pto.vmi.ensure_mask_layout cannot materialize the requested mask "
           "layout conversion ("
        << reason
        << "); partial/tail predicate layout materialization requires an "
           "explicit packing plan";
    return WalkResult::interrupt();
  }

  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIMaskGranularityOp(Operation *op) {
  if (auto ensure = dyn_cast<VMIEnsureMaskGranularityOp>(op)) {
    auto sourceType = cast<VMIMaskType>(ensure.getSource().getType());
    auto resultType = cast<VMIMaskType>(ensure.getResult().getType());
    bool identity = sourceType.getGranularity() == resultType.getGranularity() &&
                    sourceType.getLayoutAttr() == resultType.getLayoutAttr();
    if (!identity) {
      VMILayoutSupport supports;
      std::string reason;
      if (failed(supports.getMaskGranularityCastLayoutFactForLayouts(
              sourceType, resultType, sourceType.getLayoutAttr(),
              resultType.getLayoutAttr(), &reason))) {
        ensure.emitError()
            << kVMIDiagUnsupportedPrefix
            << "mask granularity cast layout relation is unsupported ("
            << reason << ")";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  }

  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMILayoutOp(Operation *op) {
  if (auto result = verifySupportedVMIEnsureLayoutOp(op);
      result.has_value()) {
    return *result;
  }
  if (auto result = verifySupportedVMIMaskLayoutOp(op); result.has_value()) {
    return *result;
  }
  return verifySupportedVMIMaskGranularityOp(op);
}

template <typename MaskCheck>
WalkResult verifySupportedCompareValue(Operation *op, StringRef opName,
                                       VMIVRegType lhsType, MaskCheck checkMaskable,
                                       LogicalResult predicateCheck) {
  WalkResult physical = checkMaskable(op, opName, lhsType);
  if (physical.wasInterrupted()) {
    return physical;
  }
  if (succeeded(predicateCheck)) {
    return WalkResult::advance();
  }
  return WalkResult::interrupt();
}

template <typename MaskCheck>
std::optional<WalkResult> verifySupportedVMICompareOp(Operation *op,
                                                      MaskCheck checkMaskable) {
  if (auto cmpf = dyn_cast<VMICmpFOp>(op)) {
    return verifySupportedCompareValue(
        op, "pto.vmi.cmpf", cast<VMIVRegType>(cmpf.getLhs().getType()),
        checkMaskable,
        checkSupportedComparePredicate<VMICmpFOp>(op, cmpf.getPredicate()));
  }

  if (auto cmpi = dyn_cast<VMICmpIOp>(op)) {
    return verifySupportedCompareValue(
        op, "pto.vmi.cmpi", cast<VMIVRegType>(cmpi.getLhs().getType()),
        checkMaskable,
        checkSupportedComparePredicate<VMICmpIOp>(op, cmpi.getPredicate()));
  }

  return std::nullopt;
}

template <typename VecScalarOp, typename MaskableCheck>
WalkResult verifySupportedVecScalarOp(VecScalarOp op, StringRef opName,
                                      MaskableCheck checkMaskable) {
  bool requiresPassthru =
      op.getPmode().has_value() && *op.getPmode() == "merge";
  if (requiresPassthru) {
    op.emitError() << kVMIDiagUnsupportedPrefix << opName
                   << " with pmode=merge requires an explicit passthru lowering";
    return WalkResult::interrupt();
  }
  return checkMaskable(op, opName,
                       cast<VMIVRegType>(op.getResult().getType()));
}

template <typename MaskableOp, typename MaskableCheck>
WalkResult verifySupportedMaskableOp(MaskableOp op, StringRef opName,
                                     MaskableCheck checkMaskable) {
  return checkMaskable(op.getOperation(), opName,
                       cast<VMIVRegType>(op.getResult().getType()));
}

WalkResult emitMaskableUnsupported(Operation *op, StringRef opName,
                                   VMIVRegType type) {
  std::string reason;
  if (succeeded(checkSupportedMaskableVReg(type, &reason))) {
    return WalkResult::advance();
  }
  op->emitError()
      << kVMIDiagUnsupportedPrefix << opName
      << " direct lowering requires physical vreg parts with b8/b16/b32 "
         "predicate masks ("
      << reason << ")";
  return WalkResult::interrupt();
}

template <typename MaskableCheck>
static std::optional<WalkResult> verifySupportedVMIUnaryBinaryArithmeticOp(
    Operation *op, MaskableCheck check) {
#define PTO_VERIFY_MASKABLE(Op, Name)                                      \
  if (auto value = dyn_cast<Op>(op)) {                                    \
    return verifySupportedMaskableOp(value, Name, check);                  \
  }
  PTO_VERIFY_MASKABLE(VMIAddFOp, "pto.vmi.addf");
  PTO_VERIFY_MASKABLE(VMIAddIOp, "pto.vmi.addi");
  PTO_VERIFY_MASKABLE(VMISubFOp, "pto.vmi.subf");
  PTO_VERIFY_MASKABLE(VMISubIOp, "pto.vmi.subi");
  PTO_VERIFY_MASKABLE(VMIMulFOp, "pto.vmi.mulf");
  PTO_VERIFY_MASKABLE(VMIMulIOp, "pto.vmi.muli");
  PTO_VERIFY_MASKABLE(VMIDivFOp, "pto.vmi.divf");
  PTO_VERIFY_MASKABLE(VMIMinFOp, "pto.vmi.minf");
  PTO_VERIFY_MASKABLE(VMIMinIOp, "pto.vmi.mini");
  PTO_VERIFY_MASKABLE(VMIMaxFOp, "pto.vmi.maxf");
  PTO_VERIFY_MASKABLE(VMIMaxIOp, "pto.vmi.maxi");
  PTO_VERIFY_MASKABLE(VMINegFOp, "pto.vmi.negf");
  PTO_VERIFY_MASKABLE(VMINegIOp, "pto.vmi.negi");
  PTO_VERIFY_MASKABLE(VMIAbsFOp, "pto.vmi.absf");
  PTO_VERIFY_MASKABLE(VMIAbsIOp, "pto.vmi.absi");
  PTO_VERIFY_MASKABLE(VMISqrtOp, "pto.vmi.sqrt");
  PTO_VERIFY_MASKABLE(VMIExpOp, "pto.vmi.exp");
  PTO_VERIFY_MASKABLE(VMILnOp, "pto.vmi.ln");
  PTO_VERIFY_MASKABLE(VMIAndIOp, "pto.vmi.andi");
  PTO_VERIFY_MASKABLE(VMIOrIOp, "pto.vmi.ori");
  PTO_VERIFY_MASKABLE(VMIXOrIOp, "pto.vmi.xori");
  PTO_VERIFY_MASKABLE(VMIShLIOp, "pto.vmi.shli");
  PTO_VERIFY_MASKABLE(VMIShRUIOp, "pto.vmi.shrui");
  PTO_VERIFY_MASKABLE(VMIShRSIOp, "pto.vmi.shrsi");
  PTO_VERIFY_MASKABLE(VMINotOp, "pto.vmi.not");
  PTO_VERIFY_MASKABLE(VMISelectOp, "pto.vmi.select");
#undef PTO_VERIFY_MASKABLE
  return std::nullopt;
}

template <typename MaskableCheck>
static std::optional<WalkResult> verifySupportedVMIVecScalarArithmeticOp(
    Operation *op, MaskableCheck check) {
  if (auto value = dyn_cast<VMIAddSOp>(op)) {
    return verifySupportedVecScalarOp(value, "pto.vmi.vadds", check);
  }
  if (auto value = dyn_cast<VMIMulSOp>(op)) {
    return verifySupportedVecScalarOp(value, "pto.vmi.vmuls", check);
  }
  if (auto value = dyn_cast<VMIMaxSOp>(op)) {
    return verifySupportedVecScalarOp(value, "pto.vmi.vmaxs", check);
  }
  if (auto value = dyn_cast<VMIMinSOp>(op)) {
    return verifySupportedVecScalarOp(value, "pto.vmi.vmins", check);
  }
  if (auto value = dyn_cast<VMIShlSOp>(op)) {
    return verifySupportedVecScalarOp(value, "pto.vmi.vshls", check);
  }
  if (auto value = dyn_cast<VMIShrSOp>(op)) {
    return verifySupportedVecScalarOp(value, "pto.vmi.vshrs", check);
  }
  return std::nullopt;
}

template <typename MaskableCheck>
std::optional<WalkResult> verifySupportedVMIArithmeticOp(Operation *op,
                                                         MaskableCheck check) {
  if (auto result = verifySupportedVMIUnaryBinaryArithmeticOp(op, check);
      result.has_value()) {
    return result;
  }
  return verifySupportedVMIVecScalarArithmeticOp(op, check);
}

template <typename ReduceOp>
WalkResult verifySupportedReduceOp(ReduceOp op, bool requiresReassoc,
                                   StringRef diagnostic) {
  std::string reason;
  if (succeeded(checkSupportedReduceShape(op, requiresReassoc, &reason))) {
    return WalkResult::advance();
  }
  op.emitError() << kVMIDiagUnsupportedPrefix << diagnostic << reason << ")";
  return WalkResult::interrupt();
}

template <typename GroupReduceOp>
WalkResult verifySupportedGroupReduceOp(GroupReduceOp op, StringRef diagnostic) {
  std::string reason;
  if (succeeded(checkSupportedGroupReduceShape(op, &reason))) {
    return WalkResult::advance();
  }
  op.emitError() << kVMIDiagUnsupportedPrefix << diagnostic << reason << ")";
  return WalkResult::interrupt();
}

template <typename ShapeOp, typename ShapeCheck>
WalkResult verifySupportedShapeOp(ShapeOp op, ShapeCheck check,
                                  StringRef diagnostic) {
  std::string reason;
  if (succeeded(check(op, &reason))) {
    return WalkResult::advance();
  }
  op.emitError() << kVMIDiagUnsupportedPrefix << diagnostic << reason << ")";
  return WalkResult::interrupt();
}

WalkResult verifySupportedConstantMaskOp(VMIConstantMaskOp op) {
  std::string reason;
  if (succeeded(computeConstantMaskMaterialization(op, &reason))) {
    return WalkResult::advance();
  }
  op.emitError()
      << kVMIDiagUnsupportedPrefix
      << "pto.vmi.constant_mask requires a dense bool constant with concrete "
         "layout and b8/b16/b32 granularity ("
      << reason << ")";
  return WalkResult::interrupt();
}

template <typename ChannelOp, typename ShapeCheck>
WalkResult verifySupportedChannelOp(ChannelOp op, int64_t channels,
                                     ShapeCheck check, StringRef supportedText,
                                     StringRef shapeText) {
  std::string reason;
  if (succeeded(check(op, &reason))) {
    return WalkResult::advance();
  }
  if (channels != 2 && channels != 4) {
    op.emitError() << kVMIDiagUnsupportedPrefix << supportedText;
  } else {
    op.emitError() << kVMIDiagUnsupportedPrefix << shapeText << reason << ")";
  }
  return WalkResult::interrupt();
}

static std::optional<WalkResult> verifySupportedVMIFloatConversionOp(
    Operation *op) {
  if (auto fptosi = dyn_cast<VMIFPToSIOp>(op)) {
    return verifySupportedShapeOp(
        fptosi, checkSupportedFPToSIShape,
        "pto.vmi.fptosi supports fp-to-signed-int conversion pairs listed in "
        "the VPTO vcvt contract; check lookupVMIFpToSiContract (");
  }
  if (auto fptoui = dyn_cast<VMIFPToUIOp>(op)) {
    return verifySupportedShapeOp(
        fptoui, checkSupportedFPToUIShape,
        "pto.vmi.fptoui supports fp-to-unsigned-int conversion pairs listed "
        "in the VPTO vcvt contract (e.g. f16 → u8); "
        "check lookupVMIFpToUIContract (");
  }
  if (auto sitofp = dyn_cast<VMISIToFPOp>(op)) {
    return verifySupportedShapeOp(
        sitofp, checkSupportedSIToFPShape,
        "pto.vmi.sitofp supports si32->f32 or si8->f16 conversion shapes (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIIntegerConversionOp(
    Operation *op) {
  if (auto extsi = dyn_cast<VMIExtSIOp>(op)) {
    return verifySupportedShapeOp(
        extsi, checkSupportedExtSIShape,
        "pto.vmi.extsi supports contiguous signed/signless 8-bit or 16-bit "
        "integer physical source chunks to 2x/4x wider integer "
        "deinterleaved results, or matching group_slots(num_groups=G, "
        "slots=1) layouts and natural group_slots(num_groups=G, slots=8, "
        "lane_stride=2/4) to group_slots(num_groups=G, slots=8) widening "
        "layouts (");
  }
  if (auto extui = dyn_cast<VMIExtUIOp>(op)) {
    return verifySupportedShapeOp(
        extui, checkSupportedExtUIShape,
        "pto.vmi.extui supports contiguous unsigned 8-bit or 16-bit integer "
        "physical source chunks to 2x/4x wider unsigned integer "
        "deinterleaved results, or matching group_slots(num_groups=G, "
        "slots=1) layouts and natural group_slots(num_groups=G, slots=8, "
        "lane_stride=2/4) to group_slots(num_groups=G, slots=8) widening "
        "layouts (");
  }
  if (auto trunci = dyn_cast<VMITruncIOp>(op)) {
    return verifySupportedShapeOp(
        trunci, checkSupportedTruncIShape,
        "pto.vmi.trunci supports integer deinterleaved source layouts whose "
        "factor is the 2x/4x narrowing multiple of the contiguous or "
        "deinterleaved result layout factor, or matching group_slots "
        "layouts and natural slots=8 narrowing layouts (");
  }
  if (auto bitcast = dyn_cast<VMIBitcastOp>(op)) {
    return verifySupportedShapeOp(
        bitcast, checkSupportedBitcastShape,
        "pto.vmi.bitcast requires matching source/result layouts with "
        "width-changing forms restricted to supported layout table rows (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIConversionOp(Operation *op) {
  if (auto result = verifySupportedVMIFloatConversionOp(op);
      result.has_value()) {
    return result;
  }
  return verifySupportedVMIIntegerConversionOp(op);
}

template <typename CarryOp, typename ShapeCheck>
std::optional<WalkResult> verifyAddCarryShape(CarryOp op, ShapeCheck check,
                                               StringRef diagnostic) {
  std::string reason;
  if (succeeded(check(op, &reason))) {
    return WalkResult::advance();
  }
  op.emitError() << kVMIDiagUnsupportedPrefix << diagnostic << reason << ")";
  return WalkResult::interrupt();
}

std::optional<WalkResult> verifySupportedVMIAddCarryOp(Operation *op) {
  if (auto addc = dyn_cast<VMIVaddcOp>(op)) {
    return verifyAddCarryShape(
        addc, checkSupportedVMIAddcShape,
        "pto.vmi.vaddc requires matching 32-bit data and b32 mask parts (");
  }
  if (auto addcs = dyn_cast<VMIVaddcsOp>(op)) {
    return verifyAddCarryShape(
        addcs, checkSupportedVMIAddcsShape,
        "pto.vmi.vaddcs requires matching 32-bit data and b32 mask parts (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIMultiplyLongOp(Operation *op) {
  if (auto vmull = dyn_cast<VMIVmullOp>(op)) {
    std::string reason;
    if (succeeded(checkSupportedVmullShape(vmull, &reason))) {
      return WalkResult::advance();
    }
    vmull.emitError()
        << kVMIDiagUnsupportedPrefix
        << "pto.vmi.vmull requires equal 64/128/256-lane i32/ui32 data "
           "ports, a matching b32 mask, and contiguous or deinterleaved "
           "factor-2/factor-4 lane_stride=1 layout ("
        << reason << ")";
    return WalkResult::interrupt();
  }
  return std::nullopt;
}

template <typename SpecialOp, typename ShapeCheck>
std::optional<WalkResult> verifySpecialUnaryShape(
    SpecialOp op, ShapeCheck check, StringRef diagnostic) {
  std::string reason;
  if (succeeded(check(op, &reason))) {
    return WalkResult::advance();
  }
  op.emitError() << kVMIDiagUnsupportedPrefix << diagnostic << reason << ")";
  return WalkResult::interrupt();
}

std::optional<WalkResult> verifySupportedVMISpecialUnaryOp(Operation *op) {
  if (auto relu = dyn_cast<VMIReluOp>(op)) {
    return verifySpecialUnaryShape(
        relu, checkSupportedReluShape,
        "pto.vmi.relu direct lowering requires physical vreg parts with b32 "
        "predicates for si32 or matching b16/b32 predicates for f16/f32 (");
  }
  if (auto vselr = dyn_cast<VMIVselrOp>(op)) {
    return verifySpecialUnaryShape(
        vselr, checkSupportedVselrShape,
        "pto.vmi.vselr supports only contiguous lane_stride=1 layouts with "
        "N=64, 128, or 256 for 8-bit, N=64 or 128 for 16-bit, or N=64 for "
        "32-bit elements (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMISpecialOp(Operation *op) {
  if (auto result = verifySupportedVMIAddCarryOp(op); result.has_value()) {
    return *result;
  }
  if (auto result = verifySupportedVMIMultiplyLongOp(op); result.has_value()) {
    return *result;
  }
  return verifySupportedVMISpecialUnaryOp(op);
}

static std::optional<WalkResult> verifySupportedVMINormalFloatReductionOp(
    Operation *op) {
  if (auto reduce = dyn_cast<VMIReduceAddFOp>(op)) {
    return verifySupportedReduceOp(
        reduce, true,
        "pto.vmi.reduce_addf lowers through pto.vcadd only with reassoc, "
        "f32 contiguous full source chunks, matching mask chunks, and one "
        "init/result chunk (");
  }
  if (auto reduce = dyn_cast<VMIReduceMaxFOp>(op)) {
    return verifySupportedReduceOp(
        reduce, false,
        "pto.vmi.reduce_maxf lowers through pto.vcmax only for f16/f32 "
        "contiguous full source chunks with matching mask chunks and one "
        "init/result chunk (");
  }
  if (auto reduce = dyn_cast<VMIReduceMinFOp>(op)) {
    return verifySupportedReduceOp(
        reduce, false,
        "pto.vmi.reduce_minf lowers through pto.vcmin only for f16/f32 "
        "contiguous full source chunks with matching mask chunks and one "
        "init/result chunk (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMINormalIntegerReductionOp(
    Operation *op) {
  if (auto reduce = dyn_cast<VMIReduceAddIOp>(op)) {
    return verifySupportedReduceOp(
        reduce, false,
        "pto.vmi.reduce_addi lowers through pto.vcadd only for contiguous "
        "full 32-bit integer source chunks with matching mask chunks and one "
        "init/result chunk (");
  }
  if (auto reduce = dyn_cast<VMIReduceMaxIOp>(op)) {
    return verifySupportedReduceOp(
        reduce, false,
        "pto.vmi.reduce_maxi lowers through pto.vcmax only for contiguous "
        "full integer source chunks with matching mask chunks and one "
        "init/result chunk (");
  }
  if (auto reduce = dyn_cast<VMIReduceMinIOp>(op)) {
    return verifySupportedReduceOp(
        reduce, false,
        "pto.vmi.reduce_mini lowers through pto.vcmin only for contiguous "
        "full integer source chunks with matching mask chunks and one "
        "init/result chunk (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMINormalReductionOp(
    Operation *op) {
  if (auto result = verifySupportedVMINormalFloatReductionOp(op);
      result.has_value()) {
    return result;
  }
  return verifySupportedVMINormalIntegerReductionOp(op);
}

static std::optional<WalkResult> verifySupportedVMIGroupFloatReductionOp(
    Operation *op) {
  if (auto reduce = dyn_cast<VMIGroupReduceAddFOp>(op)) {
    return verifySupportedGroupReduceOp(
        reduce,
        "pto.vmi.group_reduce_addf lowers through pto.vcgadd for 32B blocks "
        "or through pto.vcadd for contiguous full source/mask chunks, "
        "#pto.vmi.layout<num_groups = G, slots = K> result chunks, and "
        "num_groups deriving a group size aligned to physical chunks (");
  }
  if (auto reduce = dyn_cast<VMIGroupReduceMaxFOp>(op)) {
    return verifySupportedGroupReduceOp(
        reduce,
        "pto.vmi.group_reduce_maxf lowers through pto.vcgmax/vmax for 32B "
        "blocks or through pto.vcmax for contiguous full chunks, matching "
        "source/mask chunks, #pto.vmi.layout<num_groups = G, slots = K> "
        "result chunks, and num_groups deriving a group size aligned to "
        "physical chunks (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIGroupIntegerReductionOp(
    Operation *op) {
  if (auto reduce = dyn_cast<VMIGroupReduceAddIOp>(op)) {
    return verifySupportedGroupReduceOp(
        reduce,
        "pto.vmi.group_reduce_addi lowers through pto.vcgadd/vadd for "
        "supported 32B block classes or through an internal widening "
        "pto.vcadd path for aligned full chunks (");
  }
  if (auto reduce = dyn_cast<VMIGroupReduceMaxIOp>(op)) {
    return verifySupportedGroupReduceOp(
        reduce,
        "pto.vmi.group_reduce_maxi lowers through pto.vcgmax/vmax for "
        "supported 32B block classes or through pto.vcmax for aligned full "
        "chunks (");
  }
  if (auto reduce = dyn_cast<VMIGroupReduceMinIOp>(op)) {
    return verifySupportedGroupReduceOp(
        reduce,
        "pto.vmi.group_reduce_mini lowers through pto.vcgmin/vmin for "
        "supported 32B block classes or through pto.vcmin for aligned full "
        "chunks (");
  }
  return std::nullopt;
}

static std::optional<WalkResult> verifySupportedVMIGroupReductionOp(
    Operation *op) {
  if (auto result = verifySupportedVMIGroupFloatReductionOp(op);
      result.has_value()) {
    return result;
  }
  return verifySupportedVMIGroupIntegerReductionOp(op);
}

std::optional<WalkResult> verifySupportedVMIReductionOp(Operation *op) {
  if (auto normalResult = verifySupportedVMINormalReductionOp(op);
      normalResult.has_value()) {
    return *normalResult;
  }
  return verifySupportedVMIGroupReductionOp(op);
}

std::optional<WalkResult> verifySupportedVMIFloatOp(Operation *op) {
  if (auto fma = dyn_cast<VMIFmaOp>(op)) {
    return verifySupportedShapeOp(
        fma, checkSupportedFmaShape,
        "pto.vmi.fma lowers through pto.vmula only for f16/bf16/f32 element "
        "types (");
  }
  if (auto extf = dyn_cast<VMIExtFOp>(op)) {
    return verifySupportedShapeOp(
        extf, checkSupportedExtFShape,
        "pto.vmi.extf supports contiguous 16-bit float-like or fp8-like "
        "physical source chunks to f32 deinterleaved=2/4 results; "
        "partial/tail is allowed only when source padding maps to result "
        "padding (");
  }
  if (auto truncf = dyn_cast<VMITruncFOp>(op)) {
    return verifySupportedShapeOp(
        truncf, checkSupportedTruncFShape,
        "pto.vmi.truncf supports f32/f16/bf16 source narrowing (dense "
        "EvenOdd, Packed4, or f32 group_slots(num_groups=G, slots=1) to "
        "f16 group_slots(num_groups=G, slots=1)); non-f32 sources currently "
        "require dense contiguous layouts (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIBasicMiscOp(Operation *op) {
  if (auto constant = dyn_cast<VMIConstantOp>(op)) {
    auto denseAttr = dyn_cast<DenseElementsAttr>(constant.getValue());
    if (!denseAttr || !denseAttr.isSplat()) {
      constant.emitError()
          << kVMIDiagUnsupportedPrefix
          << "non-splat pto.vmi.constant requires a vreg immediate or "
             "scratch materialization plan";
      return WalkResult::interrupt();
    }
    return emitMaskableUnsupported(
        op, "pto.vmi.constant",
        cast<VMIVRegType>(constant.getResult().getType()));
  }
  if (auto broadcast = dyn_cast<VMIBroadcastOp>(op)) {
    return emitMaskableUnsupported(
        op, "pto.vmi.broadcast",
        cast<VMIVRegType>(broadcast.getResult().getType()));
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIHistogramOp(Operation *op) {
  if (auto broadcast = dyn_cast<VMIGroupBroadcastOp>(op)) {
    return verifySupportedShapeOp(
        broadcast, checkSupportedGroupBroadcastShape,
        "pto.vmi.group_broadcast requires #pto.vmi.layout<num_groups = G, "
        "slots = K> source, a dense full result layout, and num_groups "
        "deriving a group size that divides or is a multiple of physical "
        "chunk lanes (");
  }
  if (auto hist = dyn_cast<VMIVdhistOp>(op)) {
    return verifySupportedShapeOp(
        hist, checkSupportedVdhistShape,
        "pto.vmi.vdhist requires contiguous Nx{ui8|i8} source, contiguous "
        "b8 mask, and contiguous 256x{ui16|i16} acc/result (");
  }
  if (auto hist = dyn_cast<VMIVchistOp>(op)) {
    return verifySupportedShapeOp(
        hist, checkSupportedVchistShape,
        "pto.vmi.vchist requires contiguous Nx{ui8|i8} source, contiguous "
        "b8 mask, and contiguous 256x{ui16|i16} acc/result (");
  }
  return std::nullopt;
}

template <typename CompressionOp, typename ShapeCheck>
std::optional<WalkResult> verifyCompressionShape(
    CompressionOp op, ShapeCheck check, StringRef diagnostic) {
  std::string reason;
  if (succeeded(check(op, &reason))) {
    return WalkResult::advance();
  }
  op.emitError() << kVMIDiagUnsupportedPrefix << diagnostic << reason << ")";
  return WalkResult::interrupt();
}

std::optional<WalkResult> verifySupportedVMICompressionOp(Operation *op) {
  if (auto activePrefix = dyn_cast<VMIActivePrefixIndexOp>(op)) {
    return verifyCompressionShape(
        activePrefix, checkSupportedActivePrefixIndexShape,
        "pto.vmi.active_prefix_index lowers through pto.vusqz only for one "
        "contiguous physical chunk (");
  }
  if (auto compress = dyn_cast<VMICompressOp>(op)) {
    return verifyCompressionShape(
        compress, checkSupportedCompressShape,
        "pto.vmi.compress lowers through pto.vsqz only for one contiguous "
        "full physical chunk (");
  }
  if (auto compressStore = dyn_cast<VMICompressStoreOp>(op)) {
    return verifyCompressionShape(
        compressStore, checkSupportedCompressStoreShape,
        "pto.vmi.compress_store lowers through pto.vsqz + pto.vstur only for "
        "one contiguous full physical chunk with a UB pointer destination (");
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIMiscOp(Operation *op) {
  if (auto result = verifySupportedVMIBasicMiscOp(op); result.has_value()) {
    return *result;
  }
  if (auto result = verifySupportedVMIHistogramOp(op); result.has_value()) {
    return *result;
  }
  return verifySupportedVMICompressionOp(op);
}

std::optional<WalkResult>
verifySupportedVMIChannelShuffleOp(Operation *op) {
  if (auto split = dyn_cast<VMIChannelSplitOp>(op)) {
    return verifySupportedChannelOp(
        split, split.getNumResults(), checkSupportedChannelSplitShape,
        "pto.vmi.channel_split supports only 2 or 4 channels",
        "pto.vmi.channel_split requires source layout to be contiguous or "
        "matching deinterleaved channel layout, every result layout to be "
        "contiguous, and complete physical channel groups (");
  }
  if (auto merge = dyn_cast<VMIChannelMergeOp>(op)) {
    return verifySupportedChannelOp(
        merge, merge.getInputs().size(), checkSupportedChannelMergeShape,
        "pto.vmi.channel_merge supports only 2 or 4 channels",
        "pto.vmi.channel_merge requires every input layout to be contiguous "
        "and result layout to be contiguous or matching deinterleaved "
        "channel layout, with complete physical channel groups (");
  }
  if (auto shuffle = dyn_cast<VMIShuffleOp>(op)) {
    std::string reason;
    if (succeeded(computeShuffleForwardingSourceParts(shuffle, &reason))) {
      return WalkResult::advance();
    }
    std::string splatReason;
    if (succeeded(computeShuffleLane0SplatSourcePart(shuffle, &splatReason))) {
      return WalkResult::advance();
    }
    std::string vselrReason;
    if (succeeded(computeShuffleVselrPlans(shuffle, &vselrReason))) {
      return WalkResult::advance();
    }

    shuffle.emitError()
        << kVMIDiagUnsupportedPrefix
        << "pto.vmi.shuffle requires physical chunk forwarding or "
           "lane0 splat or vci-materializable vselr indices (forwarding: "
        << reason << "; lane0 splat: " << splatReason
        << "; vselr: " << vselrReason << ")";
    return WalkResult::interrupt();
  }
  if (auto constantMask = dyn_cast<VMIConstantMaskOp>(op)) {
    return verifySupportedConstantMaskOp(constantMask);
  }
  return std::nullopt;
}

std::optional<WalkResult> verifySupportedVMIStandardOp(
    Operation *op, bool enableStableGatherMaskedLoad) {
  if (auto memoryResult = verifySupportedVMIMemoryOp(
          op, enableStableGatherMaskedLoad);
      memoryResult.has_value()) {
    return *memoryResult;
  }
  if (auto layoutResult = verifySupportedVMILayoutOp(op);
      layoutResult.has_value()) {
    return *layoutResult;
  }
  auto compareResult = verifySupportedVMICompareOp(
      op, emitMaskableUnsupported);
  if (compareResult.has_value()) {
    return *compareResult;
  }
  if (auto miscResult = verifySupportedVMIMiscOp(op);
      miscResult.has_value()) {
    return *miscResult;
  }
  if (auto arithmeticResult = verifySupportedVMIArithmeticOp(
          op, emitMaskableUnsupported);
      arithmeticResult.has_value()) {
    return *arithmeticResult;
  }
  if (auto specialResult = verifySupportedVMISpecialOp(op);
      specialResult.has_value()) {
    return *specialResult;
  }
  if (auto reductionResult = verifySupportedVMIReductionOp(op);
      reductionResult.has_value()) {
    return *reductionResult;
  }
  if (auto floatResult = verifySupportedVMIFloatOp(op);
      floatResult.has_value()) {
    return *floatResult;
  }
  return verifySupportedVMIConversionOp(op);
}

static WalkResult verifySupportedVMIToVPTOOp(
    Operation *op, bool enableStableGatherMaskedLoad) {
  if (auto standardResult = verifySupportedVMIStandardOp(
          op, enableStableGatherMaskedLoad);
      standardResult.has_value()) {
    return *standardResult;
  }
  if (auto channelShuffleResult = verifySupportedVMIChannelShuffleOp(op);
      channelShuffleResult.has_value()) {
    return *channelShuffleResult;
  }
  return WalkResult::advance();
}

LogicalResult
verifySupportedVMIToVPTOOps(ModuleOp module,
                            bool enableStableGatherMaskedLoad) {
  WalkResult result = module.walk(
      [&enableStableGatherMaskedLoad](Operation *op) {
        return verifySupportedVMIToVPTOOp(op, enableStableGatherMaskedLoad);
      });
  return failure(result.wasInterrupted());
}



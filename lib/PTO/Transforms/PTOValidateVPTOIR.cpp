// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOValidateVPTOIR.cpp - Shared VPTO legality helpers --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file owns the shared helper layer for the dual-stage VPTO legality
// verifier. Follow-up tasks add the public validation entrypoints and pass
// wrappers on top of this utility layer.
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <type_traits>

namespace mlir {
namespace pto {

LogicalResult validateVPTOAuthoringIR(ModuleOp module,
                                      llvm::raw_ostream *diagOS = nullptr);
LogicalResult validateVPTOEmissionIR(ModuleOp module,
                                     llvm::raw_ostream *diagOS = nullptr);

namespace detail {

static Operation *getFirstNonConstantLikeOp(Block *block) {
  if (!block) {
    return nullptr;
  }
  for (Operation &op : *block) {
    if (!op.hasTrait<OpTrait::ConstantLike>()) {
      return &op;
    }
  }
  return nullptr;
}

static bool isOpInRange(Operation *op, Operation *first, Operation *last) {
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    if (cur == op) {
      return true;
    }
    if (cur == last) {
      return false;
    }
  }
  return false;
}

static constexpr int64_t kSimtKeepResumeSlotLimit = 123;
static constexpr unsigned kWideValueRegisterCount = 2;

static std::optional<unsigned> getSimtKeepResumeRegisterCount(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() <= mlir::pto::kValue32) {
      return 1;
    }
    if (intType.getWidth() == mlir::pto::kValue64) {
      return kWideValueRegisterCount;
    }
    return std::nullopt;
  }
  if (type.isF16() || type.isBF16() || type.isF32()) {
    return 1;
  }
  return std::nullopt;
}

template <typename OpT>
static Type getSimtKeepResumeValueType(OpT op);

template <>
Type getSimtKeepResumeValueType(KeepOp op) {
  return op.getPayload().getType();
}

template <>
Type getSimtKeepResumeValueType(ResumeOp op) {
  return op.getResult().getType();
}

template <typename OpT>
static LogicalResult verifySimtKeepResumeSlotRange(OpT op) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount) {
    return success();
  }
  int64_t slot = op.getSlot();
  if (slot < 0 || slot >= kSimtKeepResumeSlotLimit) {
    return op.emitOpError()
           << "requires slot in range [0, "
           << (kSimtKeepResumeSlotLimit - 1) << "]";
  }
  if (*registerCount == mlir::pto::kValue2) {
    if ((slot % kWideValueRegisterCount) != 0) {
      return op.emitOpError()
             << "requires an even slot for 64-bit keep/resume values";
    }
    if (slot + 1 >= kSimtKeepResumeSlotLimit) {
      return op.emitOpError()
             << "requires slot in range [0, "
             << (kSimtKeepResumeSlotLimit - mlir::pto::kValue2)
             << "] for 64-bit keep/resume values";
    }
  }
  return success();
}

template <typename OpT>
static bool overlapsEarlierSimtKeepResumeSlotUse(OpT op,
                                                 SmallVectorImpl<int64_t> &used) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount) {
    return false;
  }
  int64_t slot = op.getSlot();
  for (int64_t word = slot; word < slot + *registerCount; ++word) {
    if (llvm::is_contained(used, word)) {
      return true;
    }
  }
  for (int64_t word = slot; word < slot + *registerCount; ++word) {
    used.push_back(word);
  }
  return false;
}

static LogicalResult verifyUniqueResumeGroupSlots(ResumeOp current,
                                                  Operation *first) {
  SmallVector<int64_t, mlir::pto::kValue4> slots;
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    auto resume = dyn_cast<ResumeOp>(cur);
    if (!resume) {
      break;
    }
    if (overlapsEarlierSimtKeepResumeSlotUse(resume, slots) &&
        resume.getOperation() == current.getOperation()) {
      return current.emitOpError()
             << "duplicates an earlier slot " << resume.getSlot()
             << " in the SIMT resume prologue group";
    }
  }
  return success();
}

static LogicalResult verifyUniqueKeepGroupSlots(KeepOp current,
                                                Operation *first,
                                                Operation *last) {
  SmallVector<int64_t, mlir::pto::kValue4> slots;
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    auto keep = dyn_cast<KeepOp>(cur);
    if (!keep) {
      break;
    }
    if (overlapsEarlierSimtKeepResumeSlotUse(keep, slots) &&
        keep.getOperation() == current.getOperation()) {
      return current.emitOpError()
             << "duplicates an earlier slot " << keep.getSlot()
             << " in the SIMT keep epilogue group";
    }
    if (cur == last) {
      break;
    }
  }
  return success();
}

constexpr llvm::StringLiteral kAIVectorScopeAttrName =
    "llvm.loop.aivector_scope";

enum class VPTOMaskGranularity {
  B8,
  B16,
  B32,
};

enum class VPTOBufferAddressFamily {
  None,
  Copy,
  BufferLike,
  PtrOnly,
};

enum class VPTOLegalityStage {
  Authoring,
  Emission,
};

class VPTOLegalityHelper {
public:
  explicit VPTOLegalityHelper(ModuleOp module) : module(module) {}

  ModuleOp getModule() const { return module; }

  SmallVector<func::FuncOp> getFunctions() {
    SmallVector<func::FuncOp> funcs;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      funcs.push_back(func);
    }
    return funcs;
  }

  static bool isLegalityTypedValue(Type type) {
    return isa<VRegType, MaskType, AlignType>(type);
  }

  static bool isBufferLikeValue(Type type) {
    return isa<BaseMemRefType, PtrType>(type);
  }

  static bool requiresVecScope(Operation *op) {
    if (!isPTOp(op)) {
      return false;
    }

    return llvm::any_of(op->getOperandTypes(), isLegalityTypedValue) ||
           llvm::any_of(op->getResultTypes(), isLegalityTypedValue);
  }

  static bool isAIVectorScopeCarrier(scf::ForOp loop) {
    return loop && loop->hasAttr(kAIVectorScopeAttrName);
  }

  static bool isDedicatedVecScopeCarrier(Operation *op) {
    return isa_and_nonnull<VecScopeOp, StrictVecScopeOp>(op);
  }

  static bool isAnyVectorScopeCarrier(Operation *op) {
    if (auto loop = dyn_cast_or_null<scf::ForOp>(op)) {
      return isAIVectorScopeCarrier(loop);
    }
    return isDedicatedVecScopeCarrier(op);
  }

  static Operation *getEnclosingVectorScopeCarrier(Operation *op) {
    for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
         parent = parent->getParentOp()) {
      if (isAnyVectorScopeCarrier(parent)) {
        return parent;
      }
    }
    return nullptr;
  }

  static std::optional<VPTOMaskGranularity> getMaskGranularity(Type type) {
    auto maskType = dyn_cast<MaskType>(type);
    if (!maskType) {
      return std::nullopt;
    }
    return getMaskGranularity(maskType);
  }

  static std::optional<VPTOMaskGranularity> getMaskGranularity(MaskType type) {
    if (type.isB8()) {
      return VPTOMaskGranularity::B8;
    }
    if (type.isB16()) {
      return VPTOMaskGranularity::B16;
    }
    if (type.isB32()) {
      return VPTOMaskGranularity::B32;
    }
    return std::nullopt;
  }

  static StringRef stringifyMaskGranularity(VPTOMaskGranularity granularity) {
    switch (granularity) {
    case VPTOMaskGranularity::B8:
      return "b8";
    case VPTOMaskGranularity::B16:
      return "b16";
    case VPTOMaskGranularity::B32:
      return "b32";
    }
    llvm_unreachable("unsupported VPTO mask granularity");
  }

  static std::optional<VPTOMaskGranularity>
  inferMaskGranularityFromType(Type type) {
    if (auto vregType = dyn_cast<VRegType>(type)) {
      type = vregType.getElementType();
    }

    if (type.isF32()) {
      return VPTOMaskGranularity::B32;
    }
    if (type.isF16() || type.isBF16()) {
      return VPTOMaskGranularity::B16;
    }

    auto intType = dyn_cast<IntegerType>(type);
    if (!intType) {
      return std::nullopt;
    }

    switch (intType.getWidth()) {
    case mlir::pto::kValue8:
      return VPTOMaskGranularity::B8;
    case mlir::pto::kValue16:
      return VPTOMaskGranularity::B16;
    case mlir::pto::kValue32:
      return VPTOMaskGranularity::B32;
    default:
      return std::nullopt;
    }
  }

  static std::optional<VPTOMaskGranularity>
  inferMaskGranularityFromFamily(Operation *op) {
    StringRef mnemonic = getPTOpMnemonic(op);
    if (mnemonic.empty()) {
      return std::nullopt;
    }

    if (mnemonic.ends_with("_b8")) {
      return VPTOMaskGranularity::B8;
    }
    if (mnemonic.ends_with("_b16")) {
      return VPTOMaskGranularity::B16;
    }
    if (mnemonic.ends_with("_b32")) {
      return VPTOMaskGranularity::B32;
    }
    return std::nullopt;
  }

  static VPTOBufferAddressFamily classifyBufferAddressFamily(Operation *op) {
    if (!op) {
      return VPTOBufferAddressFamily::None;
    }

    if (isa<CopyGmToUbufOp, CopyUbufToUbufOp, CopyUbufToGmOp,
            CopyCbufToUbufOp, CopyUbufToCbufOp>(op)) {
      return VPTOBufferAddressFamily::Copy;
    }

    if (isa<VldasOp, VldusOp, PstuOp, VstusOp, VsturOp, MadOp, MadMxOp,
            CopyGmToCbufOp, LoadCbufToCaOp,
            LoadCbufToCbOp, CopyMatrixCcToGmOp>(op)) {
      return VPTOBufferAddressFamily::PtrOnly;
    }

    if (isa<VldsOp, UvldOp, PldsOp, PldiOp, VstsOp, PstiOp, PstsOp,
            VbitsortOp, Vmrgsort4Op, Vgather2Op,
            VgatherbOp, Vgather2BcOp, VscatterOp, Vldsx2Op, Vstsx2Op, VsldbOp,
            VsstbOp, VstasOp, VstarOp>(op)) {
      return VPTOBufferAddressFamily::BufferLike;
    }

    return VPTOBufferAddressFamily::None;
  }

  static bool isSupportedEmissionBufferLikeOp(Operation *op) {
    return classifyBufferAddressFamily(op) ==
           VPTOBufferAddressFamily::BufferLike;
  }

  static bool isResidualEmissionScaffold(Operation *op) {
    return isa<memref::SubViewOp, memref::ReinterpretCastOp,
               memref::MemorySpaceCastOp>(op) ||
           isTrivialEmissionCastPtr(op);
  }

  static SmallVector<OpOperand *> collectBufferOperands(Operation *op) {
    SmallVector<OpOperand *> bufferOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      if (isBufferLikeValue(operand.get().getType())) {
        bufferOperands.push_back(&operand);
      }
    }
    return bufferOperands;
  }

private:
  static bool isPTOp(Operation *op) {
    return op && op->getName().getStringRef().starts_with("pto.");
  }

  static StringRef getPTOpMnemonic(Operation *op) {
    if (!isPTOp(op)) {
      return {};
    }
    StringRef mnemonic = op->getName().getStringRef();
    (void)mnemonic.consume_front("pto.");
    return mnemonic;
  }

  static bool isTrivialEmissionCastPtr(Operation *op) {
    auto castOp = dyn_cast_or_null<CastPtrOp>(op);
    return castOp &&
           castOp.getInput().getType() == castOp.getResult().getType();
  }

  ModuleOp module;
};

class VPTOLegalityValidator {
public:
  VPTOLegalityValidator(ModuleOp module, VPTOLegalityStage stage,
                        llvm::raw_ostream *diagOS)
      : helper(module), stage(stage), diagOS(diagOS) {}

  LogicalResult validate() {
    if (!helper.getModule()) {
      writeDiagnostic("VPTO legality validation requires a valid module\n");
      return failure();
    }

    if (failed(validateAuthoringRules())) {
      return failure();
    }

    if (stage == VPTOLegalityStage::Emission &&
        failed(validateEmissionRules())) {
      return failure();
    }

    return success();
  }

private:
  LogicalResult validateAuthoringRules() {
    if (failed(validateAuthoringFunctionSurface())) {
      return failure();
    }
    if (failed(validateAuthoringOperationSurface())) {
      return failure();
    }
    return success();
  }

  LogicalResult validateEmissionRules() {
    if (failed(validateEmissionFunctionSurface())) {
      return failure();
    }
    if (failed(validateEmissionOperationSurface())) {
      return failure();
    }
    return success();
  }

  static std::string formatExpectedMaskType(VPTOMaskGranularity granularity) {
    std::string storage;
    llvm::raw_string_ostream os(storage);
    os << "!pto.mask<"
       << VPTOLegalityHelper::stringifyMaskGranularity(granularity) << ">";
    return storage;
  }

  static LogicalResult validateMaskMatchesVectorFamily(Operation *op,
                                                       Type maskType,
                                                       StringRef maskRole,
                                                       Type vectorType,
                                                       StringRef vectorRole) {
    auto actual = VPTOLegalityHelper::getMaskGranularity(maskType);
    auto expected = VPTOLegalityHelper::inferMaskGranularityFromType(vectorType);
    if (!actual || !expected || *actual == *expected) {
      return success();
    }

    return op->emitOpError()
           << maskRole << " " << maskType << " does not match " << vectorRole
           << " " << vectorType << "; expected "
           << formatExpectedMaskType(*expected);
  }

  static std::optional<VPTOMaskGranularity>
  inferVstsMaskGranularityOverride(Operation *op) {
    Value value;
    if (auto vsts = dyn_cast<VstsOp>(op)) {
      value = vsts.getValue();
    } else {
      return std::nullopt;
    }

    auto valueType = dyn_cast<VRegType>(value.getType());
    if (!valueType) {
      return std::nullopt;
    }

    auto distAttr = op->getAttrOfType<StringAttr>("dist");
    if (!distAttr) {
      return std::nullopt;
    }

    StringRef dist = distAttr.getValue();
    auto elementType = valueType.getElementType();
    unsigned width = 0;
    if (auto elementIntType = dyn_cast<IntegerType>(elementType)) {
      width = elementIntType.getWidth();
    } else if (elementType.isF16() || elementType.isBF16()) {
      width = mlir::pto::kValue16;
    } else if (elementType.isF32()) {
      width = mlir::pto::kValue32;
    } else if (elementType.isF64()) {
      width = mlir::pto::kValue64;
    } else {
      return std::nullopt;
    }

    if (dist == "PK_B16") {
      if (width == mlir::pto::kValue8) {
        return VPTOMaskGranularity::B16;
      }
      return std::nullopt;
    }
    if (dist == "PK_B32") {
      if (width == mlir::pto::kValue16) {
        return VPTOMaskGranularity::B32;
      }
      return std::nullopt;
    }
    if (dist == "PK_B64") {
      if (width == mlir::pto::kValue32) {
        return VPTOMaskGranularity::B32;
      }
      return std::nullopt;
    }
    if (dist == "PK4_B32") {
      if (width == mlir::pto::kValue8) {
        return VPTOMaskGranularity::B32;
      }
      return std::nullopt;
    }
    if (dist == "MRG4CHN_B8") {
      if (width == mlir::pto::kValue8) {
        return VPTOMaskGranularity::B32;
      }
      return std::nullopt;
    }
    if (dist == "MRG2CHN_B8") {
      if (width == mlir::pto::kValue8) {
        return VPTOMaskGranularity::B16;
      }
      return std::nullopt;
    }
    if (dist == "MRG2CHN_B16") {
      if (width == mlir::pto::kValue16) {
        return VPTOMaskGranularity::B32;
      }
    }
    return std::nullopt;
  }

  static LogicalResult validateSameMaskGranularity(Operation *op, Type lhsType,
                                                   StringRef lhsRole,
                                                   Type rhsType,
                                                   StringRef rhsRole) {
    auto lhs = VPTOLegalityHelper::getMaskGranularity(lhsType);
    auto rhs = VPTOLegalityHelper::getMaskGranularity(rhsType);
    if (!lhs || !rhs || *lhs == *rhs) {
      return success();
    }

    return op->emitOpError() << lhsRole << " " << lhsType << " does not match "
                             << rhsRole << " " << rhsType;
  }

  static bool isAdjacentMaskGranularityWidening(VPTOMaskGranularity input,
                                                VPTOMaskGranularity result) {
    return (input == VPTOMaskGranularity::B8 &&
            result == VPTOMaskGranularity::B16) ||
           (input == VPTOMaskGranularity::B16 &&
            result == VPTOMaskGranularity::B32);
  }

  static bool isAdjacentMaskGranularityNarrowing(VPTOMaskGranularity input,
                                                 VPTOMaskGranularity result) {
    return (input == VPTOMaskGranularity::B16 &&
            result == VPTOMaskGranularity::B8) ||
           (input == VPTOMaskGranularity::B32 &&
            result == VPTOMaskGranularity::B16);
  }

  static LogicalResult validatePpackMaskGranularity(PpackOp op) {
    auto input = VPTOLegalityHelper::getMaskGranularity(op.getInput().getType());
    auto result = VPTOLegalityHelper::getMaskGranularity(op.getResult().getType());
    if (!input || !result || *input == *result ||
        isAdjacentMaskGranularityNarrowing(*input, *result)) {
      return success();
    }

    return op.emitOpError()
           << "input mask type " << op.getInput().getType()
           << " does not match result mask type " << op.getResult().getType()
           << " for pto.ppack";
  }

  static LogicalResult validatePunpackMaskGranularity(PunpackOp op) {
    auto input = VPTOLegalityHelper::getMaskGranularity(op.getInput().getType());
    auto result = VPTOLegalityHelper::getMaskGranularity(op.getResult().getType());
    if (!input || !result || *input == *result ||
        isAdjacentMaskGranularityWidening(*input, *result)) {
      return success();
    }

    return op.emitOpError()
           << "input mask type " << op.getInput().getType()
           << " does not match result mask type " << op.getResult().getType()
           << " for pto.punpack";
  }

  template <typename OpTy>
  static LogicalResult validateInputMaskVectorConsumer(OpTy op) {
    return validateMaskMatchesVectorFamily(op, op.getMask().getType(),
                                           "mask type",
                                           op.getInput().getType(),
                                           "input vector type");
  }

  template <typename OpTy>
  static LogicalResult validateBinaryMaskVectorConsumer(OpTy op) {
    return validateMaskMatchesVectorFamily(op, op.getMask().getType(),
                                           "mask type", op.getLhs().getType(),
                                           "lhs vector type");
  }

  template <typename OpTy>
  static LogicalResult validateValueMaskVectorConsumer(OpTy op) {
    if constexpr (std::is_same_v<OpTy, VstsOp>) {
      if (std::optional<VPTOMaskGranularity> expected =
              inferVstsMaskGranularityOverride(op.getOperation())) {
        auto actual =
            VPTOLegalityHelper::getMaskGranularity(op.getMask().getType());
        if (!actual || *actual == *expected) {
          return success();
        }
        return op.emitOpError()
               << "mask type " << op.getMask().getType()
               << " does not match value vector type "
               << op.getValue().getType() << "; expected "
               << formatExpectedMaskType(*expected);
      }
    }
    return validateMaskMatchesVectorFamily(op, op.getMask().getType(),
                                           "mask type", op.getValue().getType(),
                                           "value vector type");
  }

  void emitHardwareSupportWarnings(Operation *op) const {
    auto emitForStore = [this](auto storeOp) {
      Operation *store = storeOp.getOperation();
      auto distAttr = store->getAttrOfType<StringAttr>("dist");
      if (!distAttr) {
        return;
      }

      StringRef dist = distAttr.getValue();
      if (dist == "MRG4CHN_B8" || dist == "MRG2CHN_B8" || dist == "MRG2CHN_B16") {
        writeDiagnostic((Twine("warning: ") + store->getName().getStringRef() +
                         " dist " + dist +
                         " is not supported on the current hardware\n")
                            .str());
      }
    };

    if (auto vsts = dyn_cast<VstsOp>(op)) {
      emitForStore(vsts);
      return;
    }
  }

  template <typename OpTy>
  static LogicalResult validateResultMaskVectorConsumer(OpTy op) {
    return validateMaskMatchesVectorFamily(op, op.getMask().getType(),
                                           "mask type",
                                           op.getResult().getType(),
                                           "result vector type");
  }

  template <typename CarryOp>
  static LogicalResult validateCarryFamilyContract(CarryOp op) {
    if (failed(validateMaskMatchesVectorFamily(op, op.getMask().getType(),
                                               "mask type",
                                               op.getLhs().getType(),
                                               "lhs vector type")) ||
        failed(validateSameMaskGranularity(op, op.getMask().getType(),
                                           "mask type",
                                           op.getCarry().getType(),
                                           "carry type"))) {
      return failure();
    }

    if constexpr (std::is_same_v<CarryOp, VaddcsOp> ||
                  std::is_same_v<CarryOp, VsubcsOp>) {
      if (failed(validateSameMaskGranularity(op, op.getCarryIn().getType(),
                                             "carry_in type",
                                             op.getMask().getType(),
                                             "mask type")) ||
          failed(validateSameMaskGranularity(op, op.getCarryIn().getType(),
                                             "carry_in type",
                                             op.getCarry().getType(),
                                             "carry type"))) {
        return failure();
      }
    }

    return success();
  }

  template <typename CompareOp>
  static LogicalResult validateCompareFamilyContract(CompareOp op, Type vecType) {
    if (failed(validateMaskMatchesVectorFamily(op, op.getMask().getType(),
                                               "seed mask type", vecType,
                                               "input vector type")) ||
        failed(validateMaskMatchesVectorFamily(op, op.getResult().getType(),
                                               "result mask type", vecType,
                                               "input vector type")) ||
        failed(validateSameMaskGranularity(op, op.getMask().getType(),
                                           "seed mask type",
                                           op.getResult().getType(),
                                           "result mask type"))) {
      return failure();
    }
    return success();
  }

  template <typename MaskUnaryOp>
  static LogicalResult validateMaskOnlyUnaryContract(MaskUnaryOp op) {
    return validateSameMaskGranularity(op, op.getInput().getType(),
                                       "input mask type",
                                       op.getResult().getType(),
                                       "result mask type");
  }

  static LogicalResult validateMaskOnlyPnotContract(PnotOp op) {
    if (failed(validateSameMaskGranularity(op, op.getInput().getType(),
                                           "input mask type",
                                           op.getMask().getType(),
                                           "mask type")) ||
        failed(validateSameMaskGranularity(op, op.getInput().getType(),
                                           "input mask type",
                                           op.getResult().getType(),
                                           "result mask type"))) {
      return failure();
    }
    return success();
  }

  static LogicalResult validateMaskOnlyPselContract(PselOp op) {
    if (failed(validateSameMaskGranularity(op, op.getSrc0().getType(),
                                           "src0 mask type",
                                           op.getSrc1().getType(),
                                           "src1 mask type")) ||
        failed(validateSameMaskGranularity(op, op.getSrc0().getType(),
                                           "src0 mask type",
                                           op.getMask().getType(),
                                           "mask type")) ||
        failed(validateSameMaskGranularity(op, op.getSrc0().getType(),
                                           "src0 mask type",
                                           op.getResult().getType(),
                                           "result mask type"))) {
      return failure();
    }
    return success();
  }

  template <typename PredicateMovementOp>
  static LogicalResult validatePredicateMovementContract(
      PredicateMovementOp op) {
    auto expected = VPTOLegalityHelper::inferMaskGranularityFromFamily(op);
    if (!expected) {
      return success();
    }

    if (failed(validateSameMaskGranularity(op, op.getLhs().getType(),
                                           "lhs mask type",
                                           op.getRhs().getType(),
                                           "rhs mask type")) ||
        failed(validateSameMaskGranularity(op, op.getLhs().getType(),
                                           "lhs mask type",
                                           op.getLow().getType(),
                                           "low mask type")) ||
        failed(validateSameMaskGranularity(op, op.getLhs().getType(),
                                           "lhs mask type",
                                           op.getHigh().getType(),
                                           "high mask type"))) {
      return failure();
    }

    auto lhs = VPTOLegalityHelper::getMaskGranularity(op.getLhs().getType());
    if (!lhs || *lhs == *expected) {
      return success();
    }

    return op.emitOpError()
           << "predicate movement family requires "
           << formatExpectedMaskType(*expected)
           << " but got lhs mask type " << op.getLhs().getType();
  }

  static LogicalResult validateFamilySuffixMaskResult(Operation *op,
                                                      Type resultType,
                                                      StringRef resultRole) {
    auto expected = VPTOLegalityHelper::inferMaskGranularityFromFamily(op);
    auto actual = VPTOLegalityHelper::getMaskGranularity(resultType);
    if (!expected || !actual || *expected == *actual) {
      return success();
    }

    return op->emitOpError()
           << "family suffix requires " << resultRole << " to be "
           << formatExpectedMaskType(*expected) << ", but got " << resultType;
  }

  static LogicalResult validateFamilySuffixMaskContracts(Operation *op) {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<PsetB8Op, PsetB16Op, PsetB32Op, PgeB8Op, PgeB16Op, PgeB32Op>(
            [](auto concreteOp) {
              return validateFamilySuffixMaskResult(
                  concreteOp, concreteOp.getResult().getType(), "result type");
            })
        .Case<PltB8Op, PltB16Op, PltB32Op>([](auto concreteOp) {
          return validateFamilySuffixMaskResult(concreteOp,
                                                concreteOp.getMask().getType(),
                                                "mask result type");
        })
        .Default([](Operation *) { return success(); });
  }

  static LogicalResult validateUnaryElementTypeContracts(Operation *op) {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<VreluOp>([](VreluOp concreteOp) {
          auto vecType = dyn_cast<VRegType>(concreteOp.getInput().getType());
          if (!vecType) {
            return success();
          }

          Type elemType = vecType.getElementType();
          if (auto intType = dyn_cast<IntegerType>(elemType)) {
            if (intType.getWidth() == mlir::pto::kValue32 && !intType.isUnsigned()) {
              return success();
            }
          } else if (elemType.isF16() || elemType.isF32()) {
            return success();
          }

          concreteOp.emitOpError("requires si32/i32/f16/f32 vector element type");
          return failure();
        })
        .Default([](Operation *) { return success(); });
  }

  static LogicalResult validateMaskGranularityContracts(Operation *op) {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<VabsOp, VexpOp, VlnOp, VsqrtOp, VreluOp, VnotOp,
              VcaddOp, VcmaxOp, VcminOp>(
            [](auto concreteOp) {
              return validateInputMaskVectorConsumer(concreteOp);
            })
        .Case<VaddOp, VsubOp, VmulOp, VdivOp, VmaxOp, VminOp, VandOp,
              VorOp, VxorOp, VshlOp, VshrOp>([](auto concreteOp) {
          return validateBinaryMaskVectorConsumer(concreteOp);
        })
        .Case<VaddcOp, VsubcOp, VaddcsOp, VsubcsOp>([](auto concreteOp) {
          return validateCarryFamilyContract(concreteOp);
        })
        .Case<VcmpOp>([](VcmpOp concreteOp) {
          return validateCompareFamilyContract(concreteOp,
                                               concreteOp.getSrc0().getType());
        })
        .Case<VcmpsOp>([](VcmpsOp concreteOp) {
          return validateCompareFamilyContract(concreteOp,
                                               concreteOp.getSrc().getType());
        })
        .Case<PpackOp>([](PpackOp concreteOp) {
          return validatePpackMaskGranularity(concreteOp);
        })
        .Case<PunpackOp>([](PunpackOp concreteOp) {
          return validatePunpackMaskGranularity(concreteOp);
        })
        .Case<PnotOp>(
            [](PnotOp concreteOp) { return validateMaskOnlyPnotContract(concreteOp); })
        .Case<PselOp>(
            [](PselOp concreteOp) { return validateMaskOnlyPselContract(concreteOp); })
        .Case<PdintlvB8Op, PdintlvB16Op, PdintlvB32Op,
              PintlvB8Op, PintlvB16Op, PintlvB32Op>([](auto concreteOp) {
          return validatePredicateMovementContract(concreteOp);
        })
        .Case<VselOp>([](VselOp concreteOp) {
          return validateMaskMatchesVectorFamily(concreteOp,
                                                 concreteOp.getMask().getType(),
                                                 "mask type",
                                                 concreteOp.getSrc0().getType(),
                                                 "src0 vector type");
        })
        .Case<Vgather2BcOp, VsldbOp>([](auto concreteOp) {
          return validateResultMaskVectorConsumer(concreteOp);
        })
        .Case<VstsOp, VsstbOp>([](auto concreteOp) {
          return validateValueMaskVectorConsumer(concreteOp);
        })
        .Case<Vstsx2Op>([](Vstsx2Op concreteOp) {
          return validateMaskMatchesVectorFamily(concreteOp,
                                                 concreteOp.getMask().getType(),
                                                 "mask type",
                                                 concreteOp.getLow().getType(),
                                                 "low vector type");
        })
        .Case<VmullOp, VmulaOp>([](auto concreteOp) {
          return validateMaskMatchesVectorFamily(concreteOp,
                                                 concreteOp.getMask().getType(),
                                                 "mask type",
                                                 concreteOp.getLhs().getType(),
                                                 "lhs vector type");
        })
        .Default([](Operation *) { return success(); });
  }

  LogicalResult validateAuthoringFunctionSurface() {
    for (func::FuncOp func : helper.getFunctions()) {
      auto validatePositiveI32FuncAttr =
          [&](StringRef attrName, int64_t upperBound,
              StringRef description) -> LogicalResult {
        Attribute attr = func->getAttr(attrName);
        if (!attr) {
          return success();
        }

        auto intAttr = dyn_cast<IntegerAttr>(attr);
        if (!intAttr || !intAttr.getType().isSignlessInteger(32)) {
          return func.emitError()
                 << "'" << attrName
                 << "' must be a signless i32 integer attribute";
        }

        if (intAttr.getInt() <= 0) {
          return func.emitError()
                 << "'" << attrName << "' must be a positive integer, got "
                 << intAttr.getInt();
        }

        if (intAttr.getInt() > upperBound) {
          return func.emitError()
                 << "'" << attrName << "' must be in range [1, "
                 << upperBound << "] for " << description;
        }

        if (!func->hasAttr(pto::kPTOSimtEntryAttrName)) {
          return func.emitError()
                 << "'" << attrName << "' is only allowed on functions marked '"
                 << pto::kPTOSimtEntryAttrName << "'";
        }

        return success();
      };
      if (failed(validatePositiveI32FuncAttr(pto::kPTOSimtMaxThreadsAttrName,
                                             mlir::pto::kValue2048, "SIMT max threads")) ||
          failed(validatePositiveI32FuncAttr(pto::kPTOSimtMaxRegistersAttrName,
                                             mlir::pto::kValue128, "SIMT max registers"))) {
        return failure();
      }

      if (!func->hasAttr(pto::kPTOSimtEntryAttrName)) {
        continue;
      }

      WalkResult walkResult = func.walk([&](StoreVfSimtInfoOp op) {
        op.emitOpError()
            << "must not appear inside a function marked with '"
            << pto::kPTOSimtEntryAttrName
            << "'; configure SIMT launch info in the outer non-simt caller "
               "instead";
        return WalkResult::interrupt();
      });
      if (walkResult.wasInterrupted()) {
        return failure();
      }
    }

    WalkResult keepResumeWalk = helper.getModule().walk([&](Operation *op) {
      if (!isa<KeepOp, ResumeOp, SyncthreadsOp>(op)) {
        return WalkResult::advance();
      }
      func::FuncOp func = op->getParentOfType<func::FuncOp>();
      if (!func || !func->hasAttr(pto::kPTOSimtEntryAttrName)) {
        op->emitOpError()
            << "must appear inside a function marked with '"
            << pto::kPTOSimtEntryAttrName << "'";
        return WalkResult::interrupt();
      }
      Block *block = op->getBlock();
      if (auto resume = dyn_cast<ResumeOp>(op)) {
        if (failed(verifySimtKeepResumeSlotRange(resume))) {
          return WalkResult::interrupt();
        }
        Operation *first = getFirstNonConstantLikeOp(block);
        if (!first || !isa<ResumeOp>(first)) {
          op->emitOpError()
              << "must be in the contiguous SIMT resume prologue group after "
                 "constant-like operations";
          return WalkResult::interrupt();
        }
        bool found = false;
        for (Operation *cur = first; cur; cur = cur->getNextNode()) {
          if (!isa<ResumeOp>(cur)) {
            break;
          }
          if (cur == op) {
            found = true;
            break;
          }
        }
        if (!found) {
          op->emitOpError()
              << "must be in the contiguous SIMT resume prologue group after "
                 "constant-like operations";
          return WalkResult::interrupt();
        }
        if (failed(verifyUniqueResumeGroupSlots(resume, first))) {
          return WalkResult::interrupt();
        }
      }
      if (auto keep = dyn_cast<KeepOp>(op)) {
        if (failed(verifySimtKeepResumeSlotRange(keep))) {
          return WalkResult::interrupt();
        }
        Operation *terminator = block ? block->getTerminator() : nullptr;
        if (!terminator || !isa<func::ReturnOp>(terminator)) {
          op->emitOpError()
              << "must be placed in the SIMT epilogue before func.return";
          return WalkResult::interrupt();
        }

        Operation *cur = terminator->getPrevNode();
        while (cur && isa<SyncthreadsOp>(cur)) {
          cur = cur->getPrevNode();
        }
        Operation *lastKeep = cur;
        if (!lastKeep || !isa<KeepOp>(lastKeep)) {
          op->emitOpError()
              << "must be placed in the SIMT epilogue before func.return; "
                 "only 'pto.syncthreads' may appear between the final "
                 "'pto.keep' group and func.return";
          return WalkResult::interrupt();
        }

        Operation *firstKeep = lastKeep;
        while (Operation *prev = firstKeep->getPrevNode()) {
          if (!isa<KeepOp>(prev)) {
            break;
          }
          firstKeep = prev;
        }
        if (!isOpInRange(op, firstKeep, lastKeep)) {
          op->emitOpError()
              << "must be in the contiguous SIMT keep epilogue group "
                 "immediately before optional 'pto.syncthreads' and "
                 "func.return";
          return WalkResult::interrupt();
        }
        if (failed(verifyUniqueKeepGroupSlots(keep, firstKeep, lastKeep))) {
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (keepResumeWalk.wasInterrupted()) {
      return failure();
    }
    return success();
  }

  LogicalResult validateAuthoringOperationSurface() {
    WalkResult constantWalkResult =
        helper.getModule().walk([&](arith::ConstantOp constant) {
          Type resultType = constant.getType();
          Type elementType = resultType;
          if (auto vectorType = dyn_cast<VectorType>(resultType)) {
            elementType = vectorType.getElementType();
          }
          if (!pto::isPTOFloat8Type(elementType)) {
            return WalkResult::advance();
          }

          constant.emitOpError()
              << "does not support directly constructed FP8 constants in "
                 "the VPTO backend; produce FP8 values with pto.convert";
          return WalkResult::interrupt();
        });
    if (constantWalkResult.wasInterrupted()) {
      return failure();
    }

    WalkResult loopWalkResult = helper.getModule().walk([&](scf::ForOp loop) {
      if (!VPTOLegalityHelper::isAIVectorScopeCarrier(loop)) {
        return WalkResult::advance();
      }

      Operation *parentScope =
          VPTOLegalityHelper::getEnclosingVectorScopeCarrier(loop);
      if (!parentScope) {
        return WalkResult::advance();
      }

      if (isa<scf::ForOp>(parentScope)) {
        loop.emitOpError() << "does not allow nested scf.for with '"
                           << kAIVectorScopeAttrName << "'";
        return WalkResult::interrupt();
      }

      loop.emitOpError()
          << "does not allow legacy scf.for carrier nested inside dedicated "
             "pto.vecscope/pto.strict_vecscope";
      return WalkResult::interrupt();
    });
    if (loopWalkResult.wasInterrupted()) {
      return failure();
    }

    WalkResult vecScopeWalkResult = helper.getModule().walk([&](Operation *op) {
      if (!VPTOLegalityHelper::isDedicatedVecScopeCarrier(op)) {
        return WalkResult::advance();
      }

      if (!VPTOLegalityHelper::getEnclosingVectorScopeCarrier(op)) {
        return WalkResult::advance();
      }

      op->emitOpError()
          << "does not allow nested dedicated pto.vecscope/pto.strict_vecscope";
      return WalkResult::interrupt();
    });
    if (vecScopeWalkResult.wasInterrupted()) {
      return failure();
    }

    WalkResult opWalkResult = helper.getModule().walk([&](Operation *op) {
      (void)VPTOLegalityHelper::inferMaskGranularityFromFamily(op);
      (void)VPTOLegalityHelper::classifyBufferAddressFamily(op);

      if (!VPTOLegalityHelper::requiresVecScope(op)) {
        return WalkResult::advance();
      }

      if (VPTOLegalityHelper::getEnclosingVectorScopeCarrier(op)) {
        if (failed(validateFamilySuffixMaskContracts(op)) ||
            failed(validateUnaryElementTypeContracts(op)) ||
            failed(validateMaskGranularityContracts(op))) {
          return WalkResult::interrupt();
        }
        emitHardwareSupportWarnings(op);
        return WalkResult::advance();
      }

      op->emitOpError()
          << "requires enclosing scf.for with '"
          << kAIVectorScopeAttrName
          << "' or dedicated pto.vecscope/pto.strict_vecscope"
          << "' because it consumes or produces !pto.vreg/!pto.mask/!pto.align";
      return WalkResult::interrupt();
    });
    return opWalkResult.wasInterrupted() ? failure() : success();
  }

  LogicalResult validateEmissionFunctionSurface() {
    for (func::FuncOp func : helper.getFunctions()) {
      FunctionType functionType = func.getFunctionType();

      for (auto [idx, inputType] : llvm::enumerate(functionType.getInputs())) {
        if (!isa<BaseMemRefType>(inputType)) {
          continue;
        }
        return func.emitError()
               << "emission-stage VPTO legality rejects memref argument #"
               << idx << ": " << inputType;
      }

      for (auto [idx, resultType] : llvm::enumerate(functionType.getResults())) {
        if (!isa<BaseMemRefType>(resultType)) {
          continue;
        }
        return func.emitError()
               << "emission-stage VPTO legality rejects memref result #"
               << idx << ": " << resultType;
      }
    }
    return success();
  }

  LogicalResult validateEmissionOperationSurface() {
    WalkResult walkResult = helper.getModule().walk([&](Operation *op) {
      if (isa<BuildAsyncSessionOp, TPutAsyncOp, TGetAsyncOp,
              WaitAsyncEventOp, TestAsyncEventOp>(op)) {
        op->emitOpError()
            << "is not supported by the VPTO backend; async SDMA session "
               "operations currently require the EmitC backend";
        return WalkResult::interrupt();
      }

      VPTOBufferAddressFamily family =
          VPTOLegalityHelper::classifyBufferAddressFamily(op);
      if (family == VPTOBufferAddressFamily::BufferLike) {
        for (OpOperand *operand : VPTOLegalityHelper::collectBufferOperands(op)) {
          Type operandType = operand->get().getType();
          if (!isa<BaseMemRefType>(operandType)) {
            continue;
          }

          op->emitOpError()
              << "emission-stage VPTO legality rejects memref-form buffer "
                 "operand #"
              << operand->getOperandNumber() << " of type " << operandType
              << " for buffer-like family op";
          return WalkResult::interrupt();
        }
      }

      if (VPTOLegalityHelper::isResidualEmissionScaffold(op)) {
        op->emitOpError()
            << "must be eliminated before emission-stage VPTO validation";
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });
    return walkResult.wasInterrupted() ? failure() : success();
  }

  void writeDiagnostic(StringRef message) const {
    if (diagOS) {
      *diagOS << message;
    }
  }

  VPTOLegalityHelper helper;
  VPTOLegalityStage stage;
  llvm::raw_ostream *diagOS;
};

} // namespace detail

namespace {

struct PTOValidateVPTOIRPass
    : public PassWrapper<PTOValidateVPTOIRPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOValidateVPTOIRPass)

  StringRef getArgument() const final { return "pto-validate-vpto-ir"; }

  StringRef getDescription() const final {
    return "Validate authoring-stage VPTO legality before emission-boundary canonicalization";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(validateVPTOAuthoringIR(module, &llvm::errs()))) {
      signalPassFailure();
    }
  }
};

struct PTOValidateVPTOEmissionIRPass
    : public PassWrapper<PTOValidateVPTOEmissionIRPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOValidateVPTOEmissionIRPass)

  StringRef getArgument() const final {
    return "pto-validate-vpto-emission-ir";
  }

  StringRef getDescription() const final {
    return "Validate emission-stage VPTO legality after ptr-boundary canonicalization";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(validateVPTOEmissionIR(module, &llvm::errs()))) {
      signalPassFailure();
    }
  }
};

} // namespace

LogicalResult validateVPTOAuthoringIR(ModuleOp module,
                                      llvm::raw_ostream *diagOS) {
  return detail::VPTOLegalityValidator(
             module, detail::VPTOLegalityStage::Authoring, diagOS)
      .validate();
}

LogicalResult validateVPTOEmissionIR(ModuleOp module,
                                     llvm::raw_ostream *diagOS) {
  return detail::VPTOLegalityValidator(
             module, detail::VPTOLegalityStage::Emission, diagOS)
      .validate();
}

std::unique_ptr<Pass> createPTOValidateVPTOIRPass() {
  return std::make_unique<PTOValidateVPTOIRPass>();
}

std::unique_ptr<Pass> createPTOValidateVPTOEmissionIRPass() {
  return std::make_unique<PTOValidateVPTOEmissionIRPass>();
}

} // namespace pto
} // namespace mlir

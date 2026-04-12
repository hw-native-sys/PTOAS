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

#include "PTO/IR/PTO.h"

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
    for (func::FuncOp func : module.getOps<func::FuncOp>())
      funcs.push_back(func);
    return funcs;
  }

  static bool isLegalityTypedValue(Type type) {
    return isa<VRegType, MaskType, AlignType>(type);
  }

  static bool isBufferLikeValue(Type type) {
    return isa<BaseMemRefType, PtrType>(type);
  }

  static bool requiresVecScope(Operation *op) {
    if (!isPTOp(op))
      return false;

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
    if (auto loop = dyn_cast_or_null<scf::ForOp>(op))
      return isAIVectorScopeCarrier(loop);
    return isDedicatedVecScopeCarrier(op);
  }

  static Operation *getEnclosingVectorScopeCarrier(Operation *op) {
    for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
         parent = parent->getParentOp()) {
      if (isAnyVectorScopeCarrier(parent))
        return parent;
    }
    return nullptr;
  }

  static std::optional<VPTOMaskGranularity> getMaskGranularity(Type type) {
    auto maskType = dyn_cast<MaskType>(type);
    if (!maskType)
      return std::nullopt;
    return getMaskGranularity(maskType);
  }

  static std::optional<VPTOMaskGranularity> getMaskGranularity(MaskType type) {
    if (type.isB8())
      return VPTOMaskGranularity::B8;
    if (type.isB16())
      return VPTOMaskGranularity::B16;
    if (type.isB32())
      return VPTOMaskGranularity::B32;
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
    if (auto vregType = dyn_cast<VRegType>(type))
      type = vregType.getElementType();

    if (type.isF32())
      return VPTOMaskGranularity::B32;
    if (type.isF16() || type.isBF16())
      return VPTOMaskGranularity::B16;

    auto intType = dyn_cast<IntegerType>(type);
    if (!intType)
      return std::nullopt;

    switch (intType.getWidth()) {
    case 8:
      return VPTOMaskGranularity::B8;
    case 16:
      return VPTOMaskGranularity::B16;
    case 32:
      return VPTOMaskGranularity::B32;
    default:
      return std::nullopt;
    }
  }

  static std::optional<VPTOMaskGranularity>
  inferMaskGranularityFromFamily(Operation *op) {
    StringRef mnemonic = getPTOpMnemonic(op);
    if (mnemonic.empty())
      return std::nullopt;

    if (mnemonic.ends_with("_b8"))
      return VPTOMaskGranularity::B8;
    if (mnemonic.ends_with("_b16"))
      return VPTOMaskGranularity::B16;
    if (mnemonic.ends_with("_b32"))
      return VPTOMaskGranularity::B32;
    return std::nullopt;
  }

  static VPTOBufferAddressFamily classifyBufferAddressFamily(Operation *op) {
    if (!op)
      return VPTOBufferAddressFamily::None;

    if (isa<CopyGmToUbufOp, CopyUbufToUbufOp, CopyUbufToGmOp>(op))
      return VPTOBufferAddressFamily::Copy;

    if (isa<VldsPostOp, VstsPostOp, VldasOp, VldusOp, PstuOp, VstusOp,
            VsturOp>(op))
      return VPTOBufferAddressFamily::PtrOnly;

    if (isa<VldsOp, UvldOp, PldsOp, PldiOp, VstsOp, PstiOp, PstsOp,
            VbitsortOp, Vmrgsort4Op, Vgather2Op,
            VgatherbOp, Vgather2BcOp, VscatterOp, Vldsx2Op, Vstsx2Op, VsldbOp,
            VsstbOp, VstasOp, VstarOp>(op))
      return VPTOBufferAddressFamily::BufferLike;

    return VPTOBufferAddressFamily::None;
  }

  static bool isSupportedEmissionBufferLikeOp(Operation *op) {
    return classifyBufferAddressFamily(op) ==
           VPTOBufferAddressFamily::BufferLike;
  }

  static bool isResidualEmissionScaffold(Operation *op) {
    return isa<BindTileOp, memref::SubViewOp, memref::ReinterpretCastOp,
               memref::MemorySpaceCastOp>(op) ||
           isTrivialEmissionCastPtr(op);
  }

  static SmallVector<OpOperand *> collectBufferOperands(Operation *op) {
    SmallVector<OpOperand *> bufferOperands;
    for (OpOperand &operand : op->getOpOperands()) {
      if (isBufferLikeValue(operand.get().getType()))
        bufferOperands.push_back(&operand);
    }
    return bufferOperands;
  }

private:
  static bool isPTOp(Operation *op) {
    return op && op->getName().getStringRef().starts_with("pto.");
  }

  static StringRef getPTOpMnemonic(Operation *op) {
    if (!isPTOp(op))
      return {};
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

    if (failed(validateAuthoringRules()))
      return failure();

    if (stage == VPTOLegalityStage::Emission &&
        failed(validateEmissionRules()))
      return failure();

    return success();
  }

private:
  LogicalResult validateAuthoringRules() {
    if (failed(validateAuthoringFunctionSurface()))
      return failure();
    if (failed(validateAuthoringOperationSurface()))
      return failure();
    return success();
  }

  LogicalResult validateEmissionRules() {
    if (failed(validateEmissionFunctionSurface()))
      return failure();
    if (failed(validateEmissionOperationSurface()))
      return failure();
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
    if (!actual || !expected || *actual == *expected)
      return success();

    return op->emitOpError()
           << maskRole << " " << maskType << " does not match " << vectorRole
           << " " << vectorType << "; expected "
           << formatExpectedMaskType(*expected);
  }

  static LogicalResult validateSameMaskGranularity(Operation *op, Type lhsType,
                                                   StringRef lhsRole,
                                                   Type rhsType,
                                                   StringRef rhsRole) {
    auto lhs = VPTOLegalityHelper::getMaskGranularity(lhsType);
    auto rhs = VPTOLegalityHelper::getMaskGranularity(rhsType);
    if (!lhs || !rhs || *lhs == *rhs)
      return success();

    return op->emitOpError() << lhsRole << " " << lhsType << " does not match "
                             << rhsRole << " " << rhsType;
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
    return validateMaskMatchesVectorFamily(op, op.getMask().getType(),
                                           "mask type", op.getValue().getType(),
                                           "value vector type");
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
                                           "carry type")))
      return failure();

    if constexpr (std::is_same_v<CarryOp, VaddcsOp> ||
                  std::is_same_v<CarryOp, VsubcsOp>) {
      if (failed(validateSameMaskGranularity(op, op.getCarryIn().getType(),
                                             "carry_in type",
                                             op.getMask().getType(),
                                             "mask type")) ||
          failed(validateSameMaskGranularity(op, op.getCarryIn().getType(),
                                             "carry_in type",
                                             op.getCarry().getType(),
                                             "carry type")))
        return failure();
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
                                           "result mask type")))
      return failure();
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
                                           "result mask type")))
      return failure();
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
                                           "result mask type")))
      return failure();
    return success();
  }

  template <typename PredicateMovementOp>
  static LogicalResult validatePredicateMovementContract(
      PredicateMovementOp op) {
    auto expected = VPTOLegalityHelper::inferMaskGranularityFromFamily(op);
    if (!expected)
      return success();

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
                                           "high mask type")))
      return failure();

    auto lhs = VPTOLegalityHelper::getMaskGranularity(op.getLhs().getType());
    if (!lhs || *lhs == *expected)
      return success();

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
    if (!expected || !actual || *expected == *actual)
      return success();

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

  static LogicalResult validateMaskGranularityContracts(Operation *op) {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<VabsOp, VexpOp, VlnOp, VsqrtOp, VreluOp, VnotOp,
              VcaddOp, VcmaxOp, VcminOp>(
            [](auto concreteOp) {
              return validateInputMaskVectorConsumer(concreteOp);
            })
        .Case<VaddOp, VsubOp, VmulOp, VdivOp, VmaxOp, VminOp, VandOp, VorOp,
              VxorOp, VshlOp, VshrOp>([](auto concreteOp) {
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
        .Case<PpackOp, PunpackOp>([](auto concreteOp) {
          return validateMaskOnlyUnaryContract(concreteOp);
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
        .Case<VstsOp, VstsPostOp, VsstbOp>([](auto concreteOp) {
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
      (void)func;
    }
    return success();
  }

  LogicalResult validateAuthoringOperationSurface() {
    WalkResult loopWalkResult = helper.getModule().walk([&](scf::ForOp loop) {
      if (!VPTOLegalityHelper::isAIVectorScopeCarrier(loop))
        return WalkResult::advance();

      Operation *parentScope =
          VPTOLegalityHelper::getEnclosingVectorScopeCarrier(loop);
      if (!parentScope)
        return WalkResult::advance();

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
    if (loopWalkResult.wasInterrupted())
      return failure();

    WalkResult vecScopeWalkResult = helper.getModule().walk([&](Operation *op) {
      if (!VPTOLegalityHelper::isDedicatedVecScopeCarrier(op))
        return WalkResult::advance();

      if (!VPTOLegalityHelper::getEnclosingVectorScopeCarrier(op))
        return WalkResult::advance();

      op->emitOpError()
          << "does not allow nested dedicated pto.vecscope/pto.strict_vecscope";
      return WalkResult::interrupt();
    });
    if (vecScopeWalkResult.wasInterrupted())
      return failure();

    WalkResult opWalkResult = helper.getModule().walk([&](Operation *op) {
      (void)VPTOLegalityHelper::inferMaskGranularityFromFamily(op);
      (void)VPTOLegalityHelper::classifyBufferAddressFamily(op);

      if (!VPTOLegalityHelper::requiresVecScope(op))
        return WalkResult::advance();

      if (VPTOLegalityHelper::getEnclosingVectorScopeCarrier(op))
        return (failed(validateFamilySuffixMaskContracts(op)) ||
                failed(validateMaskGranularityContracts(op)))
                   ? WalkResult::interrupt()
                   : WalkResult::advance();

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
        if (!isa<BaseMemRefType>(inputType))
          continue;
        return func.emitError()
               << "emission-stage VPTO legality rejects memref argument #"
               << idx << ": " << inputType;
      }

      for (auto [idx, resultType] : llvm::enumerate(functionType.getResults())) {
        if (!isa<BaseMemRefType>(resultType))
          continue;
        return func.emitError()
               << "emission-stage VPTO legality rejects memref result #"
               << idx << ": " << resultType;
      }
    }
    return success();
  }

  LogicalResult validateEmissionOperationSurface() {
    WalkResult walkResult = helper.getModule().walk([&](Operation *op) {
      VPTOBufferAddressFamily family =
          VPTOLegalityHelper::classifyBufferAddressFamily(op);

      if (family == VPTOBufferAddressFamily::BufferLike) {
        for (OpOperand *operand : VPTOLegalityHelper::collectBufferOperands(op)) {
          Type operandType = operand->get().getType();
          if (!isa<BaseMemRefType>(operandType))
            continue;

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
    if (diagOS)
      *diagOS << message;
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
    if (failed(validateVPTOAuthoringIR(module, &llvm::errs())))
      signalPassFailure();
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
    if (failed(validateVPTOEmissionIR(module, &llvm::errs())))
      signalPassFailure();
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

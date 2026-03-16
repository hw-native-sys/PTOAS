//===- InferPTOTileConfig.cpp - Infer arch-aware tile config -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_INFERPTOTILECONFIG
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static StringRef stringifyPTOAddressSpace(AddressSpace space) {
  switch (space) {
  case AddressSpace::Zero:
    return "zero";
  case AddressSpace::GM:
    return "gm";
  case AddressSpace::MAT:
    return "mat";
  case AddressSpace::LEFT:
    return "left";
  case AddressSpace::RIGHT:
    return "right";
  case AddressSpace::ACC:
    return "acc";
  case AddressSpace::VEC:
    return "vec";
  case AddressSpace::BIAS:
    return "bias";
  case AddressSpace::SCALING:
    return "scaling";
  }
  llvm_unreachable("unknown address space");
}

static StringRef stringifyArch(PTOArch arch) {
  return arch == PTOArch::A5 ? "a5" : "a3";
}

static std::string stringifyBLayoutValue(Attribute attr) {
  auto blayout = cast<BLayoutAttr>(attr);
  return stringifyBLayout(blayout.getValue()).str();
}

static std::string stringifySLayoutValue(Attribute attr) {
  auto slayout = cast<SLayoutAttr>(attr);
  return stringifySLayout(slayout.getValue()).str();
}

static std::string stringifyFractalValue(IntegerAttr attr) {
  return std::to_string(attr.getInt());
}

static PTOArch getTargetArch(Operation *op) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    module = op->getParentOfType<ModuleOp>();
  if (!module)
    return PTOArch::A3;
  auto arch = module->getAttrOfType<StringAttr>("pto.target_arch");
  if (arch && arch.getValue().equals_insensitive("a5"))
    return PTOArch::A5;
  return PTOArch::A3;
}

static TileBufConfigAttr inferTileConfigForSpace(MLIRContext *ctx,
                                                 AddressSpace space,
                                                 PTOArch arch,
                                                 PadValueAttr padAttr) {
  BLayout blayout = BLayout::RowMajor;
  SLayout slayout = SLayout::NoneBox;
  int32_t fractal = 512;

  switch (space) {
  case AddressSpace::LEFT:
    blayout = arch == PTOArch::A5 ? BLayout::ColMajor : BLayout::RowMajor;
    slayout = SLayout::RowMajor;
    fractal = 512;
    break;
  case AddressSpace::RIGHT:
    blayout = BLayout::RowMajor;
    slayout = SLayout::ColMajor;
    fractal = 512;
    break;
  case AddressSpace::ACC:
    blayout = BLayout::ColMajor;
    slayout = SLayout::RowMajor;
    fractal = 1024;
    break;
  default:
    return {};
  }

  Builder builder(ctx);
  if (!padAttr)
    padAttr = PadValueAttr::get(ctx, PadValue::Null);
  return TileBufConfigAttr::get(
      ctx, BLayoutAttr::get(ctx, blayout), SLayoutAttr::get(ctx, slayout),
      builder.getI32IntegerAttr(fractal), padAttr);
}

static TileBufConfigAttr mergeTileConfig(TileBufType tileTy,
                                         TileBufConfigAttr inferredConfig) {
  if (!inferredConfig)
    return {};

  auto currentConfig = tileTy.getConfigAttr();
  auto getBLayout = [&]() -> BLayoutAttr {
    if (tileTy.isConfigFieldExplicit(TileBufType::kExplicitBLayoutMask))
      return cast<BLayoutAttr>(currentConfig.getBLayout());
    return cast<BLayoutAttr>(inferredConfig.getBLayout());
  };
  auto getSLayout = [&]() -> SLayoutAttr {
    if (tileTy.isConfigFieldExplicit(TileBufType::kExplicitSLayoutMask))
      return cast<SLayoutAttr>(currentConfig.getSLayout());
    return cast<SLayoutAttr>(inferredConfig.getSLayout());
  };
  auto getFractal = [&]() -> IntegerAttr {
    if (tileTy.isConfigFieldExplicit(TileBufType::kExplicitFractalMask))
      return currentConfig.getSFractalSize();
    return inferredConfig.getSFractalSize();
  };
  auto getPad = [&]() -> PadValueAttr {
    if (tileTy.isConfigFieldExplicit(TileBufType::kExplicitPadMask))
      return cast<PadValueAttr>(currentConfig.getPad());
    return cast<PadValueAttr>(inferredConfig.getPad());
  };

  return TileBufConfigAttr::get(tileTy.getContext(), getBLayout(), getSLayout(),
                                getFractal(), getPad());
}

static TileBufConfigAttr inferMemRefTileConfig(Type memrefLikeType, PTOArch arch,
                                               MLIRContext *ctx,
                                               TileBufConfigAttr currentConfig);

static LogicalResult verifyExplicitConfigFields(
    TileBufConfigAttr currentConfig, TileBufConfigAttr expectedConfig,
    uint32_t explicitMask, AddressSpace space, PTOArch arch,
    function_ref<InFlightDiagnostic()> emitError) {
  auto emitMismatch = [&](StringRef field, StringRef expected,
                          StringRef actual) -> LogicalResult {
    auto diag = emitError();
    diag << "explicit tile config field '" << field << "' for loc="
         << stringifyPTOAddressSpace(space) << " on arch "
         << stringifyArch(arch)
         << " must be " << expected << ", got " << actual;
    return failure();
  };

  if ((explicitMask & TileBufType::kExplicitBLayoutMask) &&
      currentConfig.getBLayout() != expectedConfig.getBLayout()) {
    return emitMismatch(
        "blayout", stringifyBLayoutValue(expectedConfig.getBLayout()),
        stringifyBLayoutValue(currentConfig.getBLayout()));
  }

  if ((explicitMask & TileBufType::kExplicitSLayoutMask) &&
      currentConfig.getSLayout() != expectedConfig.getSLayout()) {
    return emitMismatch(
        "slayout", stringifySLayoutValue(expectedConfig.getSLayout()),
        stringifySLayoutValue(currentConfig.getSLayout()));
  }

  if ((explicitMask & TileBufType::kExplicitFractalMask) &&
      currentConfig.getSFractalSize() != expectedConfig.getSFractalSize()) {
    return emitMismatch(
        "fractal", stringifyFractalValue(expectedConfig.getSFractalSize()),
        stringifyFractalValue(currentConfig.getSFractalSize()));
  }

  return success();
}

static LogicalResult verifyExplicitTileBufType(
    TileBufType tileTy, PTOArch arch,
    function_ref<InFlightDiagnostic()> emitError) {
  auto spaceAttr = dyn_cast_or_null<AddressSpaceAttr>(tileTy.getMemorySpace());
  if (!spaceAttr || !tileTy.hasExplicitConfig())
    return success();

  auto expectedConfig = inferTileConfigForSpace(
      tileTy.getContext(), spaceAttr.getAddressSpace(), arch,
      dyn_cast_or_null<PadValueAttr>(tileTy.getConfigAttr().getPad()));
  if (!expectedConfig)
    return success();

  uint32_t explicitMask = tileTy.getExplicitConfigMaskValue() &
                          (TileBufType::kExplicitBLayoutMask |
                           TileBufType::kExplicitSLayoutMask |
                           TileBufType::kExplicitFractalMask);
  if (explicitMask == 0)
    return success();

  return verifyExplicitConfigFields(tileTy.getConfigAttr(), expectedConfig,
                                    explicitMask,
                                    spaceAttr.getAddressSpace(), arch,
                                    emitError);
}

static LogicalResult verifyExplicitMemRefConfig(
    Type memrefLikeType, TileBufConfigAttr currentConfig, PTOArch arch,
    function_ref<InFlightDiagnostic()> emitError) {
  if (!currentConfig)
    return success();

  auto memrefTy = dyn_cast<BaseMemRefType>(memrefLikeType);
  if (!memrefTy)
    return success();
  auto spaceAttr = dyn_cast_or_null<AddressSpaceAttr>(memrefTy.getMemorySpace());
  if (!spaceAttr)
    return success();

  auto expectedConfig = inferMemRefTileConfig(memrefLikeType, arch,
                                              memrefLikeType.getContext(),
                                              currentConfig);
  if (!expectedConfig)
    return success();

  constexpr uint32_t kRelevantExplicitMask =
      TileBufType::kExplicitBLayoutMask | TileBufType::kExplicitSLayoutMask |
      TileBufType::kExplicitFractalMask;
  return verifyExplicitConfigFields(currentConfig, expectedConfig,
                                    kRelevantExplicitMask,
                                    spaceAttr.getAddressSpace(), arch,
                                    emitError);
}

static TileBufType normalizeTileBufType(TileBufType tileTy, PTOArch arch) {
  auto spaceAttr =
      dyn_cast_or_null<AddressSpaceAttr>(tileTy.getMemorySpace());
  if (!spaceAttr)
    return {};

  auto desiredConfig = inferTileConfigForSpace(
      tileTy.getContext(), spaceAttr.getAddressSpace(), arch,
      dyn_cast_or_null<PadValueAttr>(tileTy.getConfigAttr().getPad()));
  desiredConfig = mergeTileConfig(tileTy, desiredConfig);
  if (!desiredConfig)
    return {};

  if (tileTy.hasCompleteConfig() && desiredConfig == tileTy.getConfigAttr())
    return tileTy;

  return TileBufType::get(tileTy.getContext(), tileTy.getShape(),
                          tileTy.getElementType(), tileTy.getMemorySpace(),
                          tileTy.getValidShape(), desiredConfig);
}

static TileBufConfigAttr inferMemRefTileConfig(Type memrefLikeType, PTOArch arch,
                                               MLIRContext *ctx,
                                               TileBufConfigAttr currentConfig) {
  auto memrefTy = dyn_cast<BaseMemRefType>(memrefLikeType);
  if (!memrefTy)
    return {};
  auto spaceAttr = dyn_cast_or_null<AddressSpaceAttr>(memrefTy.getMemorySpace());
  if (!spaceAttr)
    return {};
  return inferTileConfigForSpace(
      ctx, spaceAttr.getAddressSpace(), arch,
      currentConfig ? dyn_cast_or_null<PadValueAttr>(currentConfig.getPad())
                    : PadValueAttr());
}

static Type normalizeType(Type type, PTOArch arch) {
  auto tileTy = dyn_cast<TileBufType>(type);
  if (!tileTy)
    return type;
  auto normalizedTy = normalizeTileBufType(tileTy, arch);
  return normalizedTy ? Type(normalizedTy) : type;
}

static LogicalResult validateFunctionSignature(func::FuncOp func, PTOArch arch) {
  for (Type inputType : func.getFunctionType().getInputs()) {
    if (auto tileTy = dyn_cast<TileBufType>(inputType)) {
      if (failed(verifyExplicitTileBufType(
              tileTy, arch, [&]() { return func.emitOpError(); })))
        return failure();
    }
  }

  for (Type resultType : func.getFunctionType().getResults()) {
    if (auto tileTy = dyn_cast<TileBufType>(resultType)) {
      if (failed(verifyExplicitTileBufType(
              tileTy, arch, [&]() { return func.emitOpError(); })))
        return failure();
    }
  }

  return success();
}

static bool normalizeValue(Value value, PTOArch arch) {
  Type currentType = value.getType();
  Type normalizedType = normalizeType(currentType, arch);
  if (normalizedType == currentType)
    return false;
  value.setType(normalizedType);
  return true;
}

static LogicalResult syncFunctionSignature(func::FuncOp func, PTOArch arch) {
  SmallVector<Type> newInputs;
  SmallVector<Type> newResults;

  if (func.isExternal()) {
    llvm::transform(func.getArgumentTypes(), std::back_inserter(newInputs),
                    [&](Type type) { return normalizeType(type, arch); });
    llvm::transform(func.getResultTypes(), std::back_inserter(newResults),
                    [&](Type type) { return normalizeType(type, arch); });
  } else {
    Block &entry = func.front();
    newInputs.assign(entry.getArgumentTypes().begin(), entry.getArgumentTypes().end());

    if (func.getNumResults() != 0) {
      bool sawReturn = false;
      func.walk([&](func::ReturnOp ret) {
        SmallVector<Type> operandTypes(ret.getOperandTypes().begin(),
                                       ret.getOperandTypes().end());
        if (!sawReturn) {
          newResults = operandTypes;
          sawReturn = true;
          return;
        }
        if (newResults != operandTypes) {
          ret.emitOpError("all return ops must agree on result types after "
                          "tile config inference");
          func.emitError("inconsistent function result types after tile config "
                         "inference");
        }
      });
      if (!sawReturn)
        return func.emitOpError("non-external function with results must have "
                                "a return op after tile config inference");
    }
  }

  auto newFunctionType = FunctionType::get(func.getContext(), newInputs, newResults);
  if (newFunctionType != func.getFunctionType())
    func.setFunctionType(newFunctionType);
  return success();
}

static LogicalResult syncCallSites(ModuleOp module, func::FuncOp callee) {
  auto uses = callee.getSymbolUses(module);
  if (!uses)
    return success();

  for (SymbolTable::SymbolUse use : *uses) {
    auto call = dyn_cast<func::CallOp>(use.getUser());
    if (!call)
      continue;

    auto expectedInputs = callee.getFunctionType().getInputs();
    if (call.getNumOperands() != expectedInputs.size())
      return call.emitOpError("operand count does not match updated callee "
                              "signature for ")
             << callee.getSymName();

    for (auto [idx, operand] : llvm::enumerate(call.getArgOperands())) {
      if (operand.getType() != expectedInputs[idx]) {
        return call.emitOpError("operand type does not match updated callee "
                                "signature at index ")
               << idx << " for " << callee.getSymName();
      }
    }

    if (llvm::equal(call.getResultTypes(), callee.getResultTypes()))
      continue;

    OpBuilder builder(call);
    auto newCall =
        builder.create<func::CallOp>(call.getLoc(), callee, call.getArgOperands());
    newCall->setAttrs(call->getAttrs());
    call.replaceAllUsesWith(newCall.getResults());
    call.erase();
  }

  return success();
}

struct InferPTOTileConfigPass
    : public impl::InferPTOTileConfigBase<InferPTOTileConfigPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    PTOArch arch = getTargetArch(module);

    auto normalizeRegion = [&](Region &region, auto &self) -> LogicalResult {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (auto tileTy = dyn_cast<TileBufType>(arg.getType())) {
            auto *owner = arg.getOwner() ? arg.getOwner()->getParentOp() : nullptr;
            auto emitError = [&]() -> InFlightDiagnostic {
              if (owner)
                return owner->emitOpError();
              return mlir::emitError(arg.getLoc());
            };
            if (failed(verifyExplicitTileBufType(tileTy, arch, emitError)))
              return failure();
          }
          (void)normalizeValue(arg, arch);
        }

        for (Operation &op : block) {
          for (Value result : op.getResults()) {
            if (auto tileTy = dyn_cast<TileBufType>(result.getType())) {
              if (failed(verifyExplicitTileBufType(
                      tileTy, arch, [&]() { return op.emitOpError(); })))
                return failure();
            }
            (void)normalizeValue(result, arch);
          }

          if (auto pointerCast = dyn_cast<pto::PointerCastOp>(op)) {
            auto currentConfig = pointerCast.getConfig();
            if (failed(verifyExplicitMemRefConfig(
                    pointerCast.getResult().getType(),
                    currentConfig ? *currentConfig : TileBufConfigAttr(), arch,
                    [&]() { return pointerCast.emitOpError(); })))
              return failure();
            if (!currentConfig) {
              auto desiredConfig = inferMemRefTileConfig(
                  pointerCast.getResult().getType(), arch, &getContext(),
                  TileBufConfigAttr());
              if (desiredConfig)
                pointerCast->setAttr("config", desiredConfig);
            }
          }

          if (auto bindTile = dyn_cast<pto::BindTileOp>(op)) {
            if (failed(verifyExplicitMemRefConfig(
                    bindTile.getResult().getType(), bindTile.getConfig(), arch,
                    [&]() { return bindTile.emitOpError(); })))
              return failure();
          }

          for (Region &nested : op.getRegions())
            if (failed(self(nested, self)))
              return failure();
        }
      }
      return success();
    };

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (failed(validateFunctionSignature(func, arch))) {
        signalPassFailure();
        return;
      }
      if (!func.isExternal())
        if (failed(normalizeRegion(func.getBody(), normalizeRegion))) {
          signalPassFailure();
          return;
        }
      if (failed(syncFunctionSignature(func, arch))) {
        signalPassFailure();
        return;
      }
    }

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (failed(syncCallSites(module, func))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createInferPTOTileConfigPass() {
  return std::make_unique<InferPTOTileConfigPass>();
}

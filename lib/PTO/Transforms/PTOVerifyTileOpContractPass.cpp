// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOVERIFYTILEOPCONTRACT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static constexpr llvm::StringLiteral kTileOpPrimaryDomainAttr =
    "pto.tileop.primary_domain";
static constexpr llvm::StringLiteral kTileOpPhasesAttr = "pto.tileop.phases";
static constexpr llvm::StringLiteral kTileOpOperandEffectsAttr =
    "pto.tileop.operand_effects";

enum class BoundaryEffect : uint8_t {
  None,
  Read,
  Write,
  ReadWrite,
};

struct TileOpPhaseSummary {
  PIPE pipe = PIPE::PIPE_UNASSIGNED;
  llvm::SmallSet<int64_t, 8> operandUses;
  llvm::SmallSet<int64_t, 8> operandDefs;
};

template <typename CallbackT>
static void walkTileOpBodyInSourceOrder(Block &block, CallbackT &&callback) {
  for (Operation &op : block) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    callback(&op);
    if (op.getNumRegions() > 0 && op.hasTrait<OpTrait::SymbolTable>())
      continue;
    for (Region &region : op.getRegions())
      for (Block &nestedBlock : region)
        walkTileOpBodyInSourceOrder(nestedBlock, callback);
  }
}

static bool isTileOpSubkernelHelper(func::FuncOp funcOp) {
  return pto::isPTODSLTileOpHelper(funcOp);
}

static func::CallOp getTransparentWrapperCall(func::FuncOp funcOp) {
  if (!funcOp || funcOp.isDeclaration() || !funcOp.getBody().hasOneBlock())
    return nullptr;

  Block &entryBlock = funcOp.getBody().front();
  func::CallOp callOp;
  func::ReturnOp returnOp;
  for (Operation &op : entryBlock.getOperations()) {
    if (auto ret = dyn_cast<func::ReturnOp>(op)) {
      returnOp = ret;
      continue;
    }
    if (callOp)
      return nullptr;
    callOp = dyn_cast<func::CallOp>(op);
    if (!callOp)
      return nullptr;
  }

  if (!callOp || !returnOp)
    return nullptr;
  if (returnOp.getNumOperands() != callOp.getNumResults())
    return nullptr;
  for (auto [returned, forwarded] :
       llvm::zip(returnOp.getOperands(), callOp.getResults())) {
    if (returned != forwarded)
      return nullptr;
  }
  return callOp;
}

static bool isMemoryLikeBoundaryType(Type type) {
  return isa<TileBufType, TensorViewType, PartitionTensorViewType, PtrType,
             MemRefType>(type);
}

static bool isTileOpHelperBoundaryType(Type type) {
  return isa<TileBufType, TensorViewType, PartitionTensorViewType>(type);
}

static bool isTileOpScalarType(Type type) {
  return isa<IntegerType, FloatType, IndexType>(type);
}

static bool isMainVectorPipe(PIPE pipe) {
  return pipe == PIPE::PIPE_V || pipe == PIPE::PIPE_V2;
}

static bool isMainCubePipe(PIPE pipe) {
  return pipe == PIPE::PIPE_M;
}

static bool isTileOpBodyOp(Operation *op) {
  if (!op || isa<func::ReturnOp>(op))
    return false;
  if (op->getName().getDialectNamespace() != PTODialect::getDialectNamespace())
    return false;
  return true;
}

static std::optional<PIPE> getTileOpBodyPipe(Operation *op) {
  if (!isTileOpBodyOp(op))
    return std::nullopt;

  if (auto pipeOp = dyn_cast<OpPipeInterface>(op)) {
    PIPE pipe = pipeOp.getPipe();
    if (pipe != PIPE::PIPE_UNASSIGNED)
      return pipe;
  }

  if (isa<VecScopeOp, StrictVecScopeOp>(op))
    return std::nullopt;

  StringRef name = op->getName().getStringRef();
  if (name.starts_with("pto.v"))
    return PIPE::PIPE_V;
  if (name.starts_with("pto.mad"))
    return PIPE::PIPE_M;
  if (name == "pto.plt_b8" || name == "pto.plt_b16" ||
      name == "pto.plt_b32" || name == "pto.pltm_b8" ||
      name == "pto.pltm_b16" || name == "pto.pltm_b32" ||
      name == "pto.load" || name == "pto.store" || name == "pto.ldg" ||
      name == "pto.stg")
    return PIPE::PIPE_S;
  if (name == "pto.copy_gm_to_ubuf" || name == "pto.mte_gm_ub" ||
      name == "pto.mte_gm_l1" || name == "pto.mte_gm_l1_frac")
    return PIPE::PIPE_MTE2;
  if (name == "pto.mte_ub_gm" || name == "pto.mte_l0c_gm")
    return PIPE::PIPE_MTE3;
  if (name == "pto.mte_l1_l0a" || name == "pto.mte_l1_l0b" ||
      name == "pto.mte_l1_l0a_mx" || name == "pto.mte_l1_l0b_mx")
    return PIPE::PIPE_MTE1;
  return std::nullopt;
}

static FunctionKernelKind getDomainForPipe(PIPE pipe) {
  return isMainCubePipe(pipe) ? FunctionKernelKind::Cube
                              : FunctionKernelKind::Vector;
}

static StringRef stringifyBoundaryEffect(BoundaryEffect effect) {
  switch (effect) {
  case BoundaryEffect::None:
    return "none";
  case BoundaryEffect::Read:
    return "read";
  case BoundaryEffect::Write:
    return "write";
  case BoundaryEffect::ReadWrite:
    return "readwrite";
  }
  llvm_unreachable("unexpected tileop boundary effect");
}

static BoundaryEffect joinEffect(BoundaryEffect oldEffect,
                                 BoundaryEffect newEffect) {
  if (oldEffect == BoundaryEffect::None)
    return newEffect;
  if (newEffect == BoundaryEffect::None || oldEffect == newEffect)
    return oldEffect;
  return BoundaryEffect::ReadWrite;
}

static bool isSIMTOnlyPTOOp(Operation *op) {
  return isa<StoreVfSimtInfoOp, SimtLaunchOp, GetTidXOp, GetTidYOp, GetTidZOp,
             GetBlockDimXOp, GetBlockDimYOp, GetBlockDimZOp, GetGridDimXOp,
             GetGridDimYOp, GetGridDimZOp, GetBlockIdxXOp, GetBlockIdxYOp,
             GetBlockIdxZOp, GetVecCoreIdOp, GetLaneIdOp, GetClock32Op,
             GetClock64Op, GetLaneMaskEqOp, GetLaneMaskLeOp, GetLaneMaskLtOp,
             GetLaneMaskGeOp, GetLaneMaskGtOp, VoteAllOp, VoteAnyOp, VoteUniOp,
             VoteBallotOp, ShuffleIdxOp, ShuffleUpOp, ShuffleDownOp,
             ShuffleBflyOp, ReduxAddOp, ReduxMaxOp, ReduxMinOp, SyncthreadsOp,
             ThreadfenceOp, ThreadfenceBlockOp, KeepOp, ResumeOp>(op);
}

static LogicalResult buildOperandIndexMap(
    func::FuncOp funcOp, llvm::DenseMap<Value, int64_t> &operandIndex) {
  if (funcOp.isDeclaration())
    return success();
  if (!funcOp.getBody().hasOneBlock())
    return funcOp.emitOpError(
        "tileop contract verification requires a single-block helper body");

  for (auto [index, arg] :
       llvm::enumerate(funcOp.getBody().front().getArguments()))
    operandIndex.try_emplace(arg, static_cast<int64_t>(index));
  return success();
}

static Value traceBoundaryOperandToHelperArg(Value value) {
  int loopBound = 256;
  while (value && loopBound-- > 0) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto *parentOp = arg.getOwner()->getParentOp();
      if (auto forOp = dyn_cast_or_null<scf::ForOp>(parentOp)) {
        if (arg.getArgNumber() > 0 &&
            forOp.getInitArgs().size() >= arg.getArgNumber()) {
          value = forOp.getInitArgs()[arg.getArgNumber() - 1];
          continue;
        }
      }
      return value;
    }

    Operation *def = value.getDefiningOp();
    if (!def)
      return value;

    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::MemorySpaceCastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto cast = dyn_cast<memref::ReinterpretCastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
      value = collapse.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
      value = expand.getSrc();
      continue;
    }
    if (auto reshape = dyn_cast<memref::ReshapeOp>(def)) {
      value = reshape.getSource();
      continue;
    }
    if (auto transpose = dyn_cast<memref::TransposeOp>(def)) {
      value = transpose.getIn();
      continue;
    }
    if (auto view = dyn_cast<memref::ViewOp>(def)) {
      value = view.getViewSource();
      continue;
    }
    if (auto tileBufAddr = dyn_cast<TileBufAddrOp>(def)) {
      value = tileBufAddr.getSrc();
      continue;
    }
    if (auto tensorViewAddr = dyn_cast<TensorViewAddrOp>(def)) {
      value = tensorViewAddr.getSrc();
      continue;
    }
    if (auto bind = dyn_cast<BindTileOp>(def)) {
      value = bind.getSource();
      continue;
    }
    if (auto subview = dyn_cast<SubViewOp>(def)) {
      value = subview.getSource();
      continue;
    }
    if (auto bitcast = dyn_cast<BitcastOp>(def)) {
      value = bitcast.getSrc();
      continue;
    }
    if (auto reshape = dyn_cast<TReshapeOp>(def)) {
      value = reshape.getSrc();
      continue;
    }
    if (auto cast = dyn_cast<PointerCastOp>(def)) {
      if (cast.getAddrs().empty())
        return value;
      value = cast.getAddrs().front();
      continue;
    }
    if (auto cast = dyn_cast<CastPtrOp>(def)) {
      value = cast.getInput();
      continue;
    }
    if (auto addPtr = dyn_cast<AddPtrOp>(def)) {
      value = addPtr.getPtr();
      continue;
    }
    if (auto unrealized = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (unrealized.getInputs().empty())
        return value;
      if (auto result = dyn_cast<OpResult>(value)) {
        unsigned resultNumber = result.getResultNumber();
        if (resultNumber < unrealized.getInputs().size()) {
          value = unrealized.getInputs()[resultNumber];
          continue;
        }
      }
      if (unrealized.getInputs().size() == 1) {
        value = unrealized.getInputs().front();
        continue;
      }
      return value;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      if (auto result = dyn_cast<OpResult>(value)) {
        unsigned resultNumber = result.getResultNumber();
        if (resultNumber < forOp.getInitArgs().size()) {
          value = forOp.getInitArgs()[resultNumber];
          continue;
        }
      }
      return value;
    }
    return value;
  }
  return value;
}

static void recordBoundaryEffects(
    Operation *op, const llvm::DenseMap<Value, int64_t> &operandIndex,
    TileOpPhaseSummary &phase, SmallVectorImpl<BoundaryEffect> &operandEffects) {
  auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectInterface)
    return;

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 8> effects;
  effectInterface.getEffects(effects);
  for (const auto &effect : effects) {
    Value value = traceBoundaryOperandToHelperArg(effect.getValue());
    if (!value)
      continue;

    auto it = operandIndex.find(value);
    if (it == operandIndex.end())
      continue;

    int64_t index = it->second;
    if (index < 0 || index >= static_cast<int64_t>(operandEffects.size()) ||
        !isMemoryLikeBoundaryType(value.getType()))
      continue;

    BoundaryEffect boundaryEffect = BoundaryEffect::None;
    if (isa<MemoryEffects::Read>(effect.getEffect())) {
      phase.operandUses.insert(index);
      boundaryEffect = BoundaryEffect::Read;
    } else if (isa<MemoryEffects::Write>(effect.getEffect()) ||
               isa<MemoryEffects::Allocate>(effect.getEffect()) ||
               isa<MemoryEffects::Free>(effect.getEffect())) {
      phase.operandDefs.insert(index);
      boundaryEffect = BoundaryEffect::Write;
    }

    operandEffects[index] = joinEffect(operandEffects[index], boundaryEffect);
  }
}

static LogicalResult verifyScalarResults(func::FuncOp funcOp) {
  for (Type resultType : funcOp.getResultTypes()) {
    if (isTileOpScalarType(resultType))
      continue;
    return funcOp.emitOpError()
           << "tileop helper results are limited to PTO scalar values in the "
              "MVP, but found result type "
           << resultType;
  }
  return success();
}

static LogicalResult verifyArgumentBoundaryTypes(func::FuncOp funcOp) {
  for (Type argType : funcOp.getArgumentTypes()) {
    if (isTileOpHelperBoundaryType(argType) || isTileOpScalarType(argType))
      continue;
    return funcOp.emitOpError()
           << "tileop helper arguments must be Tile/TensorView/"
              "PartitionTensorView or PTO scalar values, but found "
           << argType;
  }
  return success();
}

static LogicalResult verifySummaryAttrs(func::FuncOp funcOp,
                                        std::optional<FunctionKernelKind> inferredDomain,
                                        ArrayRef<TileOpPhaseSummary> inferredPhases,
                                        ArrayRef<BoundaryEffect> inferredEffects) {
  if (!inferredDomain)
    return funcOp.emitOpError(
        "requires at least one vector or cube primary compute op; helpers with "
        "only MTE/scalar/sync phases are rejected");

  auto primaryAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(kTileOpPrimaryDomainAttr);
  if (!primaryAttr)
    return funcOp.emitOpError("requires ")
           << kTileOpPrimaryDomainAttr << " before tileop contract verification";

  if (primaryAttr.getKernelKind() != *inferredDomain) {
    return funcOp.emitOpError("has stale ")
           << kTileOpPrimaryDomainAttr << ": inferred primary domain is #pto.kernel_kind<"
           << (*inferredDomain == FunctionKernelKind::Cube ? "cube" : "vector")
           << ">";
  }

  auto phasesAttr = funcOp->getAttrOfType<ArrayAttr>(kTileOpPhasesAttr);
  if (!phasesAttr)
    return funcOp.emitOpError("requires ") << kTileOpPhasesAttr;
  if (phasesAttr.size() != inferredPhases.size()) {
    return funcOp.emitOpError("has stale ")
           << kTileOpPhasesAttr << ": inferred " << inferredPhases.size()
           << " phase(s), but attribute stores " << phasesAttr.size();
  }

  auto effectsAttr = funcOp->getAttrOfType<ArrayAttr>(kTileOpOperandEffectsAttr);
  if (!effectsAttr)
    return funcOp.emitOpError("requires ") << kTileOpOperandEffectsAttr;
  if (effectsAttr.size() != inferredEffects.size()) {
    return funcOp.emitOpError("has stale ")
           << kTileOpOperandEffectsAttr << ": inferred "
           << inferredEffects.size() << " operand effect(s), but attribute stores "
           << effectsAttr.size();
  }

  for (unsigned index = 0; index < effectsAttr.size(); ++index) {
    auto strAttr = dyn_cast<StringAttr>(effectsAttr[index]);
    if (!strAttr)
      return funcOp.emitOpError("expects ")
             << kTileOpOperandEffectsAttr
             << " entries to be string attributes";

    BoundaryEffect expected = inferredEffects[index];
    if (expected == BoundaryEffect::None)
      expected = BoundaryEffect::Read;
    if (strAttr.getValue() != stringifyBoundaryEffect(expected)) {
      return funcOp.emitOpError("has stale ")
             << kTileOpOperandEffectsAttr << " at operand #" << index
             << ": inferred \"" << stringifyBoundaryEffect(expected)
             << "\", but attribute stores \"" << strAttr.getValue() << "\"";
    }
  }

  for (unsigned index = 0; index < phasesAttr.size(); ++index) {
    auto dict = dyn_cast<DictionaryAttr>(phasesAttr[index]);
    if (!dict)
      return funcOp.emitOpError("expects ")
             << kTileOpPhasesAttr << " entries to be dictionary attributes";

    auto pipeAttr = dyn_cast_or_null<PipeAttr>(dict.get("pipe"));
    auto usesAttr = dyn_cast_or_null<ArrayAttr>(dict.get("operand_uses"));
    auto defsAttr = dyn_cast_or_null<ArrayAttr>(dict.get("operand_defs"));
    auto resultsAttr = dyn_cast_or_null<ArrayAttr>(dict.get("result_defs"));
    if (!pipeAttr || !usesAttr || !defsAttr || !resultsAttr) {
      return funcOp.emitOpError("expects ")
             << kTileOpPhasesAttr
             << " entries to carry pipe/operand_uses/operand_defs/result_defs";
    }

    const TileOpPhaseSummary &expected = inferredPhases[index];
    if (pipeAttr.getPipe() != expected.pipe) {
      return funcOp.emitOpError("has stale ")
             << kTileOpPhasesAttr << " at phase #" << index
             << ": inferred pipe " << stringifyPIPE(expected.pipe)
             << ", but attribute stores " << stringifyPIPE(pipeAttr.getPipe());
    }

    auto verifyIndexSet = [&](ArrayAttr values, llvm::SmallSet<int64_t, 8> set,
                              StringRef fieldName) -> LogicalResult {
      if (values.size() != set.size()) {
        return funcOp.emitOpError("has stale ")
               << kTileOpPhasesAttr << " at phase #" << index << " field '"
               << fieldName << "'";
      }
      llvm::SmallSet<int64_t, 8> actual;
      for (Attribute valueAttr : values) {
        auto intAttr = dyn_cast<IntegerAttr>(valueAttr);
        if (!intAttr)
          return funcOp.emitOpError("expects ") << fieldName
                                                << " indices to be integers";
        actual.insert(intAttr.getInt());
      }
      if (actual != set) {
        return funcOp.emitOpError("has stale ")
               << kTileOpPhasesAttr << " at phase #" << index << " field '"
               << fieldName << "'";
      }
      return success();
    };

    if (failed(verifyIndexSet(usesAttr, expected.operandUses, "operand_uses")) ||
        failed(verifyIndexSet(defsAttr, expected.operandDefs, "operand_defs")))
      return failure();
    if (!resultsAttr.empty()) {
      return funcOp.emitOpError("expects ")
             << kTileOpPhasesAttr
             << " result_defs to remain empty in the current MVP";
    }
  }

  return success();
}

static LogicalResult verifyTileOpHelper(func::FuncOp funcOp) {
  if (!isTileOpSubkernelHelper(funcOp) || funcOp.isDeclaration())
    return success();

  if (failed(verifyScalarResults(funcOp)) ||
      failed(verifyArgumentBoundaryTypes(funcOp)))
    return failure();

  if (auto wrapperCall = getTransparentWrapperCall(funcOp)) {
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        funcOp, wrapperCall.getCalleeAttr());
    if (callee && callee != funcOp && isTileOpSubkernelHelper(callee))
      return success();
  }

  llvm::DenseMap<Value, int64_t> operandIndex;
  if (failed(buildOperandIndexMap(funcOp, operandIndex)))
    return failure();

  SmallVector<BoundaryEffect, 8> operandEffects(funcOp.getNumArguments(),
                                                BoundaryEffect::None);
  SmallVector<TileOpPhaseSummary, 8> phases;
  std::optional<FunctionKernelKind> primaryDomain;

  Block &entry = funcOp.getBody().front();
  LogicalResult walkResult = success();
  walkTileOpBodyInSourceOrder(entry, [&](Operation *op) {
    if (failed(walkResult))
      return;

    if (isa<AllocTileOp, ReserveBufferOp, TAllocOp>(op)) {
      walkResult = op->emitError("is not allowed inside a tileop helper; "
                                 "tileop helpers must not allocate "
                                 "helper-local tile or reserved-buffer state");
      return;
    }

    if (isSIMTOnlyPTOOp(op)) {
      walkResult =
          op->emitError("is SIMT-only and cannot appear inside a tileop helper");
      return;
    }

    if (auto callOp = dyn_cast<func::CallOp>(op)) {
      if (callOp->getParentOfType<func::FuncOp>() != funcOp)
        return;
      auto module = funcOp->getParentOfType<ModuleOp>();
      auto callee = module ? module.lookupSymbol<func::FuncOp>(callOp.getCallee())
                           : func::FuncOp();
      if (callee && isTileOpSubkernelHelper(callee)) {
        InFlightDiagnostic diag =
            callOp.emitOpError("cannot call tileop helper @");
        diag << callee.getSymName()
             << " from another tileop helper; nested tileop calls are "
                "rejected";
        walkResult = failure();
        return;
      }
      return;
    }

    if (op->getName().getDialectNamespace() != PTODialect::getDialectNamespace())
      return;

    std::optional<PIPE> maybePipe = getTileOpBodyPipe(op);
    if (!maybePipe)
      return;
    PIPE pipe = *maybePipe;

    if (isMainVectorPipe(pipe) || isMainCubePipe(pipe)) {
      FunctionKernelKind domain = getDomainForPipe(pipe);
      if (primaryDomain && *primaryDomain != domain) {
        walkResult = op->emitError(
            "tileop helper mixes vector and cube primary compute pipes; MVP "
            "supports exactly one primary domain");
        return;
      }
      primaryDomain = domain;
    }

    if (phases.empty() || phases.back().pipe != pipe) {
      TileOpPhaseSummary phase;
      phase.pipe = pipe;
      phases.push_back(std::move(phase));
    }
    recordBoundaryEffects(op, operandIndex, phases.back(), operandEffects);
  });

  if (failed(walkResult))
    return failure();

  return verifySummaryAttrs(funcOp, primaryDomain, phases, operandEffects);
}

struct PTOVerifyTileOpContractPass
    : public mlir::pto::impl::PTOVerifyTileOpContractBase<
          PTOVerifyTileOpContractPass> {
  void runOnOperation() override {
    if (failed(verifyTileOpHelper(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVerifyTileOpContractPass() {
  return std::make_unique<PTOVerifyTileOpContractPass>();
}
